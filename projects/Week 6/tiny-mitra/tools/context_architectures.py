"""Matched source dependence across A, B, bare multi-head and full Tiny-GPT.

All four trained primary checkpoints see the same source pair and supplied
prefix. This records actual choices/probabilities, including failures.
"""
import hashlib
import json
from pathlib import Path
import torch
from generate import generate
from task_runtime import load_task_checkpoint, task_source_hash
from task_tokenizer import TaskTokenizer
from train import atomic_json


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    root = Path("artifacts/mitra_final")
    output = root / "results/investigations"
    names = ["embedding_only", "single_attention", "multi_attention", "transformer_2"]
    prefix = "PATIENT [NAME] RECEIVED "
    sources = [f"PATIENT John Smith RECEIVED Drug{letter}" for letter in "AB"]
    protocol = {"architectures": names, "seed": 42, "updates": 4000, "sources": sources, "conditional_prefix": prefix, "selection": "primary fixed-update checkpoints; no training or model selection from this probe", "runner_source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "source_hash": task_source_hash()}
    registration = output / "architecture_context_protocol.json"
    if registration.exists() and json.loads(registration.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Context protocol differs; preserve and review it")
    atomic_json(registration, protocol)
    tokenizer = TaskTokenizer()
    inputs = [tokenizer.encode(tokenizer.prompt(source) + prefix) for source in sources]
    assert len(inputs[0]) == len(inputs[1]) and inputs[0][-1] == inputs[1][-1]
    changed = [index for index, pair in enumerate(zip(*inputs)) if pair[0] != pair[1]]
    assert len(changed) == 1
    rows = []
    for name in names:
        path = root / f"checkpoints/{name}_seed42/latest.pt"
        model, saved_tokenizer, checkpoint = load_task_checkpoint(path)
        assert checkpoint["step"] == 4000 and checkpoint["protocol"]["seed"] == 42 and model.config.architecture == name
        assert saved_tokenizer.fingerprint == tokenizer.fingerprint and checkpoint["source_hash"] == task_source_hash()
        probabilities, observations = [], []
        for source, ids in zip(sources, inputs):
            logits = model(torch.tensor([ids]))[0, -1]
            probability = torch.softmax(logits, dim=-1)
            probabilities.append(probability)
            actual = generate(model, tokenizer, source)
            observations.append({"source": source, "input_ids": ids, "selected_token": tokenizer.tokens[int(logits.argmax())], "candidates": {piece: float(probability[tokenizer.token_to_id[piece]]) for piece in ("DrugA", "DrugB")}, "free_generation": actual})
        identical = torch.equal(probabilities[0], probabilities[1])
        if name == "embedding_only": assert identical
        correct = all(row["selected_token"] == f"Drug{letter}" and row["free_generation"]["text"] == f"PATIENT [NAME] RECEIVED Drug{letter}" and row["free_generation"]["protocol_valid"] for row, letter in zip(observations, "AB"))
        rows.append({"architecture": name, "checkpoint_step": 4000, "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "dataset_hash": checkpoint["dataset_hash"], "source_hash": checkpoint["source_hash"], "tokenizer_hash": tokenizer.fingerprint, "identical_distributions": identical, "maximum_probability_change": float((probabilities[1] - probabilities[0]).abs().max()), "both_conditional_and_free_correct": correct, "observations": observations})
    report = {"protocol": protocol, "changed_token_positions": changed, "matched_query_index": len(inputs[0]) - 1, "models": rows, "scope": "one authored development source pair; a supplied prefix is distinct from free generation; no broad generalization or significance claim"}
    atomic_json(output / "architecture_context.json", report)
    for row in rows: print(row["architecture"], [item["selected_token"] for item in row["observations"]], row["both_conditional_and_free_correct"])


if __name__ == "__main__":
    main()
