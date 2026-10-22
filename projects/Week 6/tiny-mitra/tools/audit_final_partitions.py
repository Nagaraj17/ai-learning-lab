"""Recompute final held-out counts from raw strings and check every identity.

Run after tools.final_partitions. No training or checkpoint selection occurs.
"""
import hashlib
import json
from pathlib import Path
import torch
from assignment_evaluate import counts_for_note
from config import ARCHITECTURES
from data import digest
from evaluate import aggregate
from task_runtime import load_task_checkpoint, task_source_hash
from train import atomic_json


def main():
    torch.set_num_threads(2)
    root = Path("artifacts/mitra_final")
    records = json.loads((root / "dataset/examples.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "results/protocol.json").read_text(encoding="utf-8"))
    review = json.loads(Path("reports/FINAL_PROTOCOL_REVIEW.json").read_text(encoding="utf-8"))
    identity = {"dataset_hash": digest(records), "source_hash": task_source_hash(), "protocol_hash": digest(protocol), "data_producer_source_hash": protocol["data_producer_source_hash"]}
    assert review["status"] == "passed" and review["no_test_tuning"]
    assert all(review[key] == value for key, value in identity.items())
    summary = json.loads((root / "results/final_test/summary.json").read_text(encoding="utf-8"))
    splits = sorted({row["split"] for row in records if row["split"].startswith("test_")})
    assert len(summary) == 32 and len(splits) == 4
    assert {(row["architecture"], row["split"]) for row in summary} == {(name, split) for name in ARCHITECTURES for split in splits}
    checks = []
    for architecture in ARCHITECTURES:
        checkpoint_path = root / f"checkpoints/{architecture}_seed42/latest.pt"
        model, tokenizer, checkpoint = load_task_checkpoint(checkpoint_path)
        assert model.config.architecture == architecture and checkpoint["step"] == protocol["steps"]
        assert checkpoint["dataset_hash"] == identity["dataset_hash"] and checkpoint["source_hash"] == identity["source_hash"]
        checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        for split in splits:
            selected = {row["id"]: row for row in records if row["split"] == split}
            report = json.loads((root / f"results/final_test/{architecture}_{split}.json").read_text(encoding="utf-8"))
            assert len(selected) == 50 and sorted(raw["id"] for raw in report["outputs"]) == sorted(selected)
            assert all(report[key] == value for key, value in identity.items())
            assert report["checkpoint_sha256"] == checkpoint_sha and report["tokenizer_hash"] == tokenizer.fingerprint
            counts = []
            for raw in report["outputs"]:
                expected = selected[raw["id"]]
                assert raw["source"] == expected["source"] and raw["expected"] == expected["target"]
                measured = counts_for_note(expected, raw["generated"])
                assert measured == raw["counts"]
                counts.append(measured)
            measured = aggregate(counts)
            row = next(row for row in summary if row["architecture"] == architecture and row["split"] == split)
            assert all(report["metrics"][key] == value and row[key] == value for key, value in measured.items())
            assert report["metrics"]["valid_terminations"] == sum(raw["protocol_valid"] for raw in report["outputs"])
            checks.append({"architecture": architecture, "split": split, "notes": 50, "passed": True})
    atomic_json(root / "results/final_test/audit.json", {"passed": True, "checks": checks, "model_note_evaluations": 1600, "distinct_notes": 200, "scope": "author audit of saved identities and recomputed counts; no training or test-based selection"})
    print("Audited all32 frozen final partitions:1600 evaluations over200distinct notes")


if __name__ == "__main__":
    main()
