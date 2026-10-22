"""Representation fidelity and resumable training for the separate adapter."""
import pytest
import torch
from assignment_data import create_dataset
from config import ModelConfig
from data import render, training_item
from task_runtime import load_task_checkpoint, train_task
from task_tokenizer import TaskTokenizer
from tokenizer import CharacterTokenizer
from train import load_checkpoint


def test_fixed_pieces_retain_spelling_offsets_and_shorter_task_sequences():
    task, chars = TaskTokenizer(), CharacterTokenizer()
    text = "PATIENT Olivia Martinez MRN 991723 DIAGNOSIS NSCLC"
    ids, offsets = task.encode_with_offsets(text)
    assert task.decode(ids) == text
    assert all(text[start:end] == task.tokens[index] for index, (start, end) in zip(ids, offsets))
    assert len(task.encode("PATIENT")) == 1
    assert task.decode(task.encode("Maya DrugAstrange")) == "Maya DrugAstrange"
    assert len(ids) < len(chars.encode(text))
    assert "Olivia Martinez" not in task.tokens and "991723" not in task.tokens
    assert task.fingerprint != chars.fingerprint
    for record in create_dataset():
        inputs, targets = training_item(record, task, 256)
        assert targets[-1] == task.eos_id and len(inputs) == len(targets)
    with pytest.raises(ValueError, match="Unsupported character"): task.encode("Zoë")


def test_task_token_runtime_exact_resume_and_loader_identity(tmp_path):
    torch.set_num_threads(2)
    records = []
    for index, (name, drug) in enumerate((("Jo", "DrugA"), ("Li", "DrugB"), ("Xi", "DrugA"))):
        record = render([("PATIENT", name), ("TREATMENT", drug)])
        record.update(id=str(index), split="validation" if index == 2 else "train")
        records.append(record)
    config = ModelConfig(len(TaskTokenizer()), d_model=16, d_ff=32, context=128)
    options = dict(seed=11, batch_size=2, validate_every=2)
    whole, _, _ = train_task(config, records, tmp_path / "whole", steps=4, **options)
    train_task(config, records, tmp_path / "resumed", steps=2, **options)
    resumed, _, _ = train_task(config, records, tmp_path / "resumed", steps=4, resume=True, **options)
    assert all(torch.equal(value, resumed.state_dict()[name]) for name, value in whole.state_dict().items())
    _, tokenizer, saved = load_task_checkpoint(tmp_path / "whole/latest.pt")
    assert saved["step"] == 4 and tokenizer.metadata()["type"].startswith("fixed_task_words")
    with pytest.raises(ValueError, match="Not a Tiny Mitra"): load_checkpoint(tmp_path / "whole/best.pt")
    with pytest.raises(ValueError, match="Resume data/source/config/protocol differs"):
        train_task(config, records, tmp_path / "resumed", steps=6, seed=12, batch_size=2, validate_every=2, resume=True)
