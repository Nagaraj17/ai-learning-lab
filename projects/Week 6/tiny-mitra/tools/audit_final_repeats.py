"""Audit all final-corpus repeat runs and their raw development counts."""
import json
import hashlib
from pathlib import Path
import torch
from assignment_data import instructor_examples
from assignment_evaluate import counts_for_note
from data import digest
from evaluate import aggregate
from experiments import diagnostic_subset
from task_runtime import load_task_checkpoint, task_source_hash
from train import atomic_json


def main():
    torch.set_num_threads(2)
    root = Path("artifacts/mitra_final")
    records = json.loads((root / "dataset/examples.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "results/repeats_protocol.json").read_text(encoding="utf-8"))
    rows = json.loads((root / "results/repeats_summary.json").read_text(encoding="utf-8"))
    assert len(rows) == 8 and protocol["source_hash"] == task_source_hash() and protocol["dataset_hash"] == digest(records)
    assert hashlib.sha256(Path("submission_repeats.py").read_bytes()).hexdigest() == protocol["runner_source_hash"]
    assert {(row["architecture"], row["seed"]) for row in rows} == {(name, seed) for name in protocol["architectures"] for seed in protocol["seeds"]}
    checks = []
    for architecture in protocol["architectures"]:
        for seed in protocol["seeds"]:
            model, tokenizer, checkpoint = load_task_checkpoint(root / f"checkpoints/{architecture}_seed{seed}/latest.pt")
            assert model.config.architecture == architecture and checkpoint["protocol"]["seed"] == seed and checkpoint["step"] == 4000
            assert checkpoint["source_hash"] == task_source_hash() and checkpoint["dataset_hash"] == digest(records)
            for kind, selected in (("validation", [row for row in records if row["split"] == "validation"]), ("acceptance", instructor_examples())):
                if kind == "validation": selected = diagnostic_subset(selected, 24)
                report = json.loads((root / f"results/{architecture}_seed{seed}_{kind}.json").read_text(encoding="utf-8"))
                assert report["seed"] == seed and report["selected_step"] == 4000 and report["dataset_hash"] == digest(records)
                assert report["tokenizer_hash"] == tokenizer.fingerprint and report["source_hash"] == checkpoint["source_hash"]
                lookup, counts = {row["id"]: row for row in selected}, []
                assert sorted(raw["id"] for raw in report["outputs"]) == sorted(lookup), "Duplicate or missing diagnostic note"
                for raw in report["outputs"]:
                    record = lookup[raw["id"]]
                    assert raw["source"] == record["source"] and raw["expected"] == record["target"]
                    measured = counts_for_note(record, raw["generated"])
                    assert measured == raw["counts"]
                    counts.append(measured)
                measured = aggregate(counts)
                assert measured["examples"] == (24 if kind == "validation" else 11)
                assert report["metrics"]["valid_terminations"] == sum(raw["protocol_valid"] for raw in report["outputs"])
                for key, value in measured.items(): assert report["metrics"][key] == value
            checks.append({"architecture": architecture, "seed": seed, "updates": 4000, "passed": True})
    atomic_json(root / "results/repeats_audit.json", {"passed": True, "checks": checks, "scope": "training-seed repeats vary initialization and minibatch order; shared development notes are not independent datasets; primary seed42 is unchanged"})
    print("Audited eight final-corpus repeat runs")


if __name__ == "__main__":
    main()
