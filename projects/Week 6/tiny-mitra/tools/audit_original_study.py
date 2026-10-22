"""Audit saved original-study identities and recompute counts from raw text.

This checks existing evidence; it does not retrain, tune on final partitions or
turn a weak generated answer into a correct one.
"""
import json
import math
from pathlib import Path
import torch
from config import ARCHITECTURES
from data import digest, load_dataset
from evaluate import aggregate, note_counts
from train import atomic_json, load_checkpoint, source_hash


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_raw(report, records):
    indexed = {record["id"]: record for record in records}
    counts = []
    assert len({row["id"] for row in report["outputs"]}) == len(report["outputs"])
    for row in report["outputs"]:
        record = indexed[row["id"]]
        assert row["source"] == record["source"] and row["expected"] == record["target"]
        measured = note_counts(record, row["generated"])
        assert measured == row["counts"]
        counts.append(measured)
    recomputed = aggregate(counts)
    for key, value in recomputed.items(): assert report["metrics"][key] == value, key
    return recomputed


def main():
    torch.set_num_threads(2)
    records = load_dataset()
    data_hash, core_hash = digest(records), source_hash()
    primary, repeats = read("artifacts/results/protocol.json"), read("artifacts/results/protocol_repeats.json")
    review = read("reports/PROTOCOL_REVIEW.json")
    assert review["status"] == "passed" and review["protocol_hash"] == digest(primary)
    for protocol in (primary, repeats):
        assert protocol["dataset_hash"] == data_hash and protocol["source_hash"] == core_hash
        assert protocol["steps"] == 1000 and protocol["generation_subset_limit"] == 12
    repeated = read("artifacts/results/repeats_summary.json")
    final = read("artifacts/results/final_test/summary.json")
    assert len(repeated) == 8 and len(final) == 32
    repeat_checks, final_checks = [], []
    for architecture in repeats["architectures"]:
        for seed in repeats["seeds"]:
            run = f"{architecture}_seed{seed}"
            directory = Path("artifacts/checkpoints") / run
            metadata = read(directory / "manifest.json")
            model, tokenizer, checkpoint = load_checkpoint(directory / "best.pt")
            assert checkpoint["step"] == metadata["selected_step"] and metadata["completed_steps"] == 1000
            assert model.config.architecture == architecture and checkpoint["protocol"]["seed"] == seed
            assert checkpoint["source_hash"] == core_hash and checkpoint["dataset_hash"] == data_hash
            assert all(math.isfinite(row["validation"]["loss"]) and math.isfinite(row["gradient_norm"]) for row in checkpoint["history"])
            report = read(f"artifacts/results/{run}_validation.json")
            assert report["checkpoint_step"] == checkpoint["step"] and report["tokenizer_hash"] == tokenizer.fingerprint
            assert report["source_hash"] == core_hash and report["dataset_hash"] == data_hash
            measured = check_raw(report, [row for row in records if row["split"] == "validation"])
            assert measured["examples"] == 12
            summary = next(row for row in repeated if row["architecture"] == architecture and row["seed"] == seed)
            for key in ("recall", "exact_match", "clinical_preservation"): assert summary[key] == measured[key]
            repeat_checks.append({"run": run, "selected_step": checkpoint["step"], "raw_notes": 12, "passed": True})
    for architecture in ARCHITECTURES:
        _, tokenizer, checkpoint = load_checkpoint(f"artifacts/checkpoints/{architecture}_seed42/best.pt")
        for split in sorted({row["split"] for row in records if row["split"].startswith("test_")}):
            selected = [row for row in records if row["split"] == split]
            report = read(f"artifacts/results/final_test/{architecture}_{split}.json")
            assert report["checkpoint_step"] == checkpoint["step"] and report["source_hash"] == core_hash
            assert report["dataset_hash"] == data_hash and report["tokenizer_hash"] == tokenizer.fingerprint
            assert {row["id"] for row in report["outputs"]} == {row["id"] for row in selected}
            measured = check_raw(report, selected)
            assert measured["examples"] == 50
            summary = next(row for row in final if row["architecture"] == architecture and row["split"] == split)
            for key, value in measured.items(): assert summary[key] == value
            final_checks.append({"architecture": architecture, "partition": split, "raw_notes": 50, "exact": measured["exact"], "passed": True})
    investigation_checks = []
    for path in sorted(Path("artifacts/results/investigations").glob("*.json")):
        report = read(path)
        if not isinstance(report, dict) or "outputs" not in report or "metrics" not in report: continue
        # Qualitative OOD files intentionally have no gold labels/metric counts.
        if not report["outputs"] or "counts" not in report["outputs"][0]: continue
        measured = check_raw(report, [row for row in records if row["split"] == "validation"])
        assert report["source_hash"] == core_hash and report["dataset_hash"] == data_hash
        assert report["checkpoint_step"] == 400 and measured["examples"] == 8
        investigation_checks.append({"file": path.name, "notes": 8, "passed": True})
    result = {"passed": True, "scope": "author engineering audit; existing outputs/identities/counts checked without retraining or changing the frozen models; not an independent reviewer", "source_hash": core_hash, "dataset_hash": data_hash, "repeat_checks": repeat_checks, "final_checks": final_checks, "investigation_checks": investigation_checks, "final_generation_evaluations": 1600, "distinct_final_notes": 200}
    atomic_json("artifacts/results/completed_original_study_audit.json", result)
    print(f"Passed: {len(repeat_checks)} repeat runs, {len(final_checks)} final partitions and {len(investigation_checks)} investigation reports; raw counts recomputed")


if __name__ == "__main__":
    main()
