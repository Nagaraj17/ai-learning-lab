"""Audit saved final-model head/pair/temperature counts and query identities."""
import hashlib
import json
from pathlib import Path
import torch
from assignment_evaluate import counts_for_note
from data import digest
from evaluate import aggregate
from task_runtime import load_task_checkpoint
from train import atomic_json


def main():
    torch.set_num_threads(2)
    root = Path("artifacts/mitra_final")
    output = root / "results/investigations"
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    records = json.loads((root / "dataset/examples.json").read_text(encoding="utf-8"))
    indexed = {row["id"]: row for row in records if row["split"] == "validation"}
    model, tokenizer, checkpoint = load_task_checkpoint(root / "checkpoints/transformer_2_seed42/latest.pt")
    assert model.config.architecture == "transformer_2" and checkpoint["step"] == protocol["checkpoint_step"] == 4000
    assert protocol["diagnostic_source_hash"] == hashlib.sha256(Path("tools/investigate_task.py").read_bytes()).hexdigest()
    assert checkpoint["dataset_hash"] == protocol["dataset_hash"] == digest(records)
    assert protocol["source_hash"] == checkpoint["source_hash"] and protocol["tokenizer_hash"] == tokenizer.fingerprint
    names = ["baseline"] + [f"layer{layer}_head{head}" for layer in range(2) for head in range(4)] + ["pair_layer0", "pair_layer1"] + [f"temperature_{temperature}" for temperature in (0.1, 0.7, 1.2)]
    checked = []
    for name in names:
        report = json.loads((output / f"{name}.json").read_text(encoding="utf-8"))
        assert sorted(raw["id"] for raw in report["outputs"]) == sorted(protocol["example_ids"])
        assert len(report["outputs"]) == 8
        assert all(report[key] == protocol[key] for key in ("run_id", "checkpoint_step", "dataset_hash", "source_hash", "tokenizer_hash"))
        counts = []
        for raw in report["outputs"]:
            record = indexed[raw["id"]]
            assert raw["source"] == record["source"] and raw["expected"] == record["target"]
            measured = counts_for_note(record, raw["generated"])
            assert measured == raw["counts"]
            counts.append(measured)
        measured = aggregate(counts)
        assert all(report["generated_metrics"][key] == value for key, value in measured.items())
        assert report["generated_metrics"]["valid_terminations"] == sum(raw["protocol_valid"] for raw in report["outputs"])
        assert report["teacher_forced"]["scored_tokens"] > 0 and report["teacher_forced"]["loss"] >= 0
        checked.append(name)
    atomic_json(output / "audit.json", {"passed": True, "checked_raw_generation_reports": checked, "diagnostic_notes_per_report": 8, "scope": "saved identities/count recomputation; intervention restoration was checked by the runner; semantic head roles are not inferred"})
    print("Audited14 final-model baseline/head/pair/temperature reports")


if __name__ == "__main__":
    main()
