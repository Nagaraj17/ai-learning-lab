"""Audit the earlier corrected-grammar character study without relabelling it."""
import json
from pathlib import Path
import torch
from assignment_evaluate import counts_for_note
from config import ARCHITECTURES
from data import digest
from evaluate import aggregate
from experiments import diagnostic_subset
from train import atomic_json, load_checkpoint, source_hash


def main():
    from assignment_data import instructor_examples
    torch.set_num_threads(2)
    root = Path("artifacts/assignment_v2")
    records = json.loads((root / "dataset/examples.json").read_text(encoding="utf-8"))
    rows = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
    assert len(rows) == 8 and {row["architecture"] for row in rows} == set(ARCHITECTURES)
    checks = []
    for architecture in ARCHITECTURES:
        directory = root / f"checkpoints/{architecture}_seed42"
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        model, tokenizer, checkpoint = load_checkpoint(directory / "best.pt")
        assert model.config.architecture == architecture and model.config.context == 256
        assert manifest["completed_steps"] == 2000 and manifest["selected_step"] == checkpoint["step"]
        assert checkpoint["source_hash"] == source_hash() and checkpoint["dataset_hash"] == digest(records)
        for kind, selected in (("validation", [row for row in records if row["split"] == "validation"]), ("acceptance", instructor_examples())):
            if kind == "validation": selected = diagnostic_subset(selected, 24)
            report = json.loads((root / f"results/{architecture}_{kind}.json").read_text(encoding="utf-8"))
            assert report["selected_step"] == checkpoint["step"] and report["dataset_hash"] == checkpoint["dataset_hash"]
            assert report["source_hash"] == checkpoint["source_hash"]
            lookup = {row["id"]: row for row in selected}
            assert sorted(raw["id"] for raw in report["outputs"]) == sorted(lookup), "Duplicate or missing diagnostic note"
            counts = []
            for raw in report["outputs"]:
                record = lookup[raw["id"]]
                assert raw["source"] == record["source"] and raw["expected"] == record["target"]
                measured = counts_for_note(record, raw["generated"])
                assert measured == raw["counts"]
                counts.append(measured)
            measured = aggregate(counts)
            for key, value in measured.items(): assert report["metrics"][key] == value
            assert measured["examples"] == (24 if kind == "validation" else 11)
            assert report["metrics"]["valid_terminations"] == sum(raw["protocol_valid"] for raw in report["outputs"])
        checks.append({"architecture": architecture, "updates": 2000, "selected_step": checkpoint["step"], "passed": True})
    atomic_json(root / "results/audit.json", {"passed": True, "scope": "earlier corrected-grammar character study; identities/raw counts verified, not a quality endorsement or final submission", "checks": checks})
    print("Audited all eight earlier corrected-grammar character runs")


if __name__ == "__main__":
    main()
