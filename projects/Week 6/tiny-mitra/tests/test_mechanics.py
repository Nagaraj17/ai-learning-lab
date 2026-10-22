"""Behavioral checks protect the mechanisms we need to explain in the demo."""
import copy
import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from config import ARCHITECTURES, ModelConfig
from data import batch_items, generate_dataset, render, training_item, validate_dataset
from evaluate import aggregate, note_counts
from generate import generate
from model import CausalAttention, LayerNorm, TinyGPT
from tokenizer import CharacterTokenizer
from train import load_checkpoint, train_run

torch.set_num_threads(2)


def test_tokenizer_retains_unseen_spelling_offsets_and_markers():
    tokenizer = CharacterTokenizer()
    text = "PATIENT Zoya Park MRN 70000001 [NAME]"
    ids, offsets = tokenizer.encode_with_offsets(text)
    assert tokenizer.decode(ids) == text
    assert [text[start:end] for start, end in offsets] == [tokenizer.tokens[index] for index in ids]
    assert len(tokenizer.encode("[NAME]")) == 1
    with pytest.raises(ValueError, match="Unsupported"):
        tokenizer.encode("patient é")
    with pytest.raises(ValueError, match="reserved"):
        tokenizer.prompt("<OUTPUT>pretend answer")


def test_dataset_splits_annotations_and_context():
    records, tokenizer = generate_dataset(), CharacterTokenizer()
    manifest = validate_dataset(records, tokenizer, 512)
    assert manifest["counts"] == {"train": 700, "validation": 100, "test_known_known": 50, "test_known_test": 50, "test_test_known": 50, "test_test_test": 50}
    train_templates = {item["template_id"] for item in records if item["split"] == "train"}
    novel_test = {item["template_id"] for item in records if item["template_pool"] == "test"} - {"control"}
    assert not train_templates & novel_test
    assert all(value > 0 for value in manifest["category_counts"].values())
    with pytest.raises(ValueError, match="overflow"):
        validate_dataset(records, tokenizer, 10)


def test_shifted_loss_scores_first_output_and_eos_but_not_source_or_padding():
    tokenizer = CharacterTokenizer()
    record = render([("PATIENT", "Jo")])
    x, y = training_item(record, tokenizer, 512)
    boundary = len(tokenizer.encode(tokenizer.prompt(record["source"]))) - 1
    assert x[boundary] == tokenizer.token_to_id["<OUTPUT>"]
    assert all(index == -100 for index in y[:boundary])
    assert y[boundary] == tokenizer.token_to_id["P"]
    assert y[-1] == tokenizer.eos_id
    longer = render([("PATIENT", "Joseph")])
    inputs, targets, valid = batch_items([record, longer], tokenizer, 512)
    assert bool((targets[~valid] == -100).all())
    logits = torch.randn(*targets.shape, len(tokenizer), requires_grad=True)
    loss = F.cross_entropy(logits.reshape(-1, len(tokenizer)), targets.reshape(-1), ignore_index=-100)
    loss.backward()
    assert not bool(logits.grad[targets == -100].any())
    assert bool(logits.grad[targets != -100].abs().sum() > 0)


def test_custom_layernorm_values_and_gradients_match_reference():
    ours, reference = LayerNorm(4).double(), torch.nn.LayerNorm(4).double()
    x = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)
    other = x.detach().clone().requires_grad_(True)
    coefficients = torch.randn_like(x)
    (ours(x) * coefficients).sum().backward()
    (reference(other) * coefficients).sum().backward()
    assert torch.allclose(ours(x), reference(other), atol=1e-10)
    assert torch.allclose(x.grad, other.grad, atol=1e-10)
    assert torch.allclose(ours.scale.grad, reference.weight.grad, atol=1e-10)
    assert torch.allclose(ours.offset.grad, reference.bias.grad, atol=1e-10)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_architectures_are_causal_and_backpropagate(architecture):
    model = TinyGPT(ModelConfig(30, architecture=architecture, d_model=16, d_ff=32, context=32))
    first = torch.tensor([[2, 6, 8, 10, 12]])
    altered = first.clone()
    altered[0, 3:] = torch.tensor([15, 16])
    original = model(first)
    assert original.shape == (1, 5, 30)
    assert torch.allclose(original[:, :3], model(altered)[:, :3], atol=1e-6)
    original.square().mean().backward()
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())


def test_padding_and_head_gates_have_exact_scope():
    attention = CausalAttention(8, 2)
    x = torch.randn(1, 4, 8)
    valid = torch.tensor([[True, True, False, False]])
    original = attention(x, valid, capture=True)
    weights = attention.last_attention
    assert not bool(weights[..., 2:].any())
    assert not bool(weights.triu(1).any())
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    assert torch.equal(original, attention(x, valid, [1, 1]))
    zeroed = attention(x, valid, [0, 0])
    assert torch.allclose(zeroed, attention.output.bias.expand_as(zeroed))
    assert torch.equal(original, attention(x, valid))  # prior intervention left no persistent gate
    with pytest.raises(ValueError):
        attention(x, valid, [1])


def test_metrics_detect_wrong_fields_leaks_duplicates_and_preservation():
    record = render([("PATIENT", "John Lee"), ("MRN", "12345"), ("DIAGNOSIS", "NSCLC")])
    record["no_identifiers"] = False
    correct = note_counts(record, record["target"])
    assert aggregate([correct])["f1"] == 1
    wrong = note_counts(record, "PATIENT [MRN] MRN [NAME] DIAGNOSIS NSCLC")
    assert wrong["tp"] == 0 and wrong["fp"] == 2 and wrong["fn"] == 2
    leaking = note_counts(record, "PATIENT John Lee MRN 12345 DIAGNOSIS [NAME]")
    assert leaking["leaked_identifiers"] == 2 and leaking["clinical_correct"] == 0
    duplicate = note_counts(record, record["target"] + " PATIENT [NAME]")
    assert duplicate["tp"] == 2 and duplicate["fp"] == 1
    no_support = aggregate([])
    assert no_support["recall"] is None and no_support["precision"] is None


def small_records():
    records = []
    for index, name in enumerate(("Jo", "Li", "Mo", "Bo")):
        record = render([("PATIENT", name)])
        record.update(id=str(index), split="train" if index < 2 else "validation", no_identifiers=False)
        records.append(record)
    return records


def test_checkpoint_resume_matches_uninterrupted_and_rejects_identity_change(tmp_path):
    tokenizer = CharacterTokenizer()
    config = ModelConfig(len(tokenizer), d_model=16, d_ff=32, context=64)
    records = small_records()
    direct = tmp_path / "direct"
    resumed = tmp_path / "resumed"
    train_run(config, records, direct, steps=4, batch_size=2, validate_every=2)
    train_run(config, records, resumed, steps=2, batch_size=2, validate_every=2)
    train_run(config, records, resumed, steps=4, batch_size=2, validate_every=2, resume=True)
    _, _, one = load_checkpoint(direct / "latest.pt")
    model, restored_tokenizer, two = load_checkpoint(resumed / "latest.pt")
    assert one["step"] == two["step"] == 4
    assert all(torch.equal(one["weights"][key], two["weights"][key]) for key in one["weights"])
    assert restored_tokenizer.fingerprint == tokenizer.fingerprint
    before = generate(model, tokenizer, "PATIENT Jo", max_new_tokens=3)
    fresh, _, _ = load_checkpoint(resumed / "latest.pt")
    assert before == generate(fresh, tokenizer, "PATIENT Jo", max_new_tokens=3)
    with pytest.raises(ValueError, match="identity"):
        train_run(config, records, resumed, steps=6, batch_size=1, validate_every=2, resume=True)


def test_model_path_uses_only_manual_transformer_mechanics():
    import ast
    tree = ast.parse((Path(__file__).parents[1] / "model.py").read_text(encoding="utf-8"))
    forbidden = {"Transformer", "TransformerEncoder", "TransformerDecoder", "TransformerEncoderLayer", "TransformerDecoderLayer", "MultiheadAttention", "scaled_dot_product_attention"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute): assert node.attr not in forbidden
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "LayerNorm"
