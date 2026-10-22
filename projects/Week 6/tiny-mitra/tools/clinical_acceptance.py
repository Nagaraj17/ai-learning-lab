"""Explicit checks for the assignment's preserved clinical terms.

The frozen eleven-prompt suite already covers NSCLC, EGFR and DrugA/B. Add a
Chemotherapy continuation and a no-identifier clinical control without changing
the training corpus, tokenizer or eleven-prompt experiment identities.
"""
import json
import argparse
from pathlib import Path
import torch
from generate import generate
from task_runtime import load_task_checkpoint
from train import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path("artifacts/mitra_final")
    model, tokenizer, checkpoint = load_task_checkpoint(root / f"checkpoints/transformer_2_seed{args.seed}/latest.pt")
    protocol = json.loads((root / "results/protocol.json").read_text(encoding="utf-8"))
    if checkpoint["step"] != protocol["steps"]: raise ValueError("Wait for the final model budget")
    cases = [
        ("PATIENT John Smith RECEIVED Chemotherapy", "PATIENT [NAME] RECEIVED Chemotherapy"),
        ("DIAGNOSIS NSCLC BIOMARKER EGFR TREATMENT Chemotherapy", "DIAGNOSIS NSCLC BIOMARKER EGFR TREATMENT Chemotherapy"),
    ]
    registration = root / "results/clinical_acceptance_protocol.json"
    declared = {"scope": "two authored clinical development checks, not final-test estimates", "cases": [{"source": source, "expected": expected} for source, expected in cases]}
    if registration.exists() and json.loads(registration.read_text(encoding="utf-8")) != declared: raise ValueError("Clinical acceptance registration differs")
    atomic_json(registration, declared)
    training = {row["source"] for row in json.loads((root / "dataset/examples.json").read_text(encoding="utf-8")) if row["split"] == "train"}
    rows = []
    for source, expected in cases:
        actual = generate(model, tokenizer, source)
        rows.append({"source": source, "expected": expected, "generated": actual["text"], "termination": actual["termination"], "seen_exact_source_in_training": source in training, "passed": actual["text"] == expected and actual["protocol_valid"]})
    report = {"passed": all(row["passed"] for row in rows), "training_seed": args.seed, "checkpoint_step": checkpoint["step"], "source_hash": checkpoint["source_hash"], "dataset_hash": checkpoint["dataset_hash"], "scope": declared["scope"], "outputs": rows}
    name = "clinical_acceptance.json" if args.seed == 42 else f"clinical_acceptance_seed{args.seed}.json"
    atomic_json(root / "results" / name, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
