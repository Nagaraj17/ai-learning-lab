"""Measure the already selected development demo on all frozen final partitions.

Selection must already be fixed from development evidence. These outcomes must
never be used to choose a different seed or edit the reported trained model.
"""
import hashlib
import json
from pathlib import Path
import torch
from assignment_evaluate import counts_for_note, generated_metrics_v2
from data import digest
from evaluate import aggregate, token_metrics
from task_runtime import load_task_checkpoint
from train import atomic_json


def main():
    torch.set_num_threads(2)
    root = Path("artifacts/mitra_final")
    selection = json.loads(Path("reports/DEMO_SELECTION.json").read_text(encoding="utf-8"))
    if not selection["passed"]:
        atomic_json(root / "results/demo_final_test/summary.json", {"status": "no passing development candidate; none selected from final outcomes", "rows": []})
        return
    assert selection["protocol"]["no_final_test_selection"]
    assert json.loads((root / "results/final_test/audit.json").read_text(encoding="utf-8"))["passed"]
    candidate = selection["selected"]
    path = Path(candidate["checkpoint"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == candidate["checkpoint_sha256"]
    model, tokenizer, checkpoint = load_task_checkpoint(path)
    records = json.loads((root / "dataset/examples.json").read_text(encoding="utf-8"))
    assert checkpoint["step"] == 4000 and checkpoint["protocol"]["seed"] == candidate["seed"]
    assert checkpoint["dataset_hash"] == digest(records)
    output = root / "results/demo_final_test"
    output.mkdir(parents=True, exist_ok=True)
    identity = {"selection_hash": digest(selection), "checkpoint_sha256": candidate["checkpoint_sha256"], "dataset_hash": digest(records), "source_hash": checkpoint["source_hash"], "tokenizer_hash": tokenizer.fingerprint, "training_seed": candidate["seed"], "checkpoint_step": 4000}
    rows = []
    for split in sorted({row["split"] for row in records if row["split"].startswith("test_")}):
        selected = [row for row in records if row["split"] == split]
        assert len(selected) == 50
        target = output / f"seed{candidate['seed']}_{split}.json"
        if target.exists():
            report = json.loads(target.read_text(encoding="utf-8"))
            assert all(report[key] == value for key, value in identity.items())
        else:
            report = generated_metrics_v2(model, selected, tokenizer)
            report.update(identity, split=split, teacher_forced=token_metrics(model, selected, tokenizer), scope="already development-selected demo; no seed changes from these outcomes")
            atomic_json(target, report)
        lookup = {row["id"]: row for row in selected}
        assert sorted(raw["id"] for raw in report["outputs"]) == sorted(lookup)
        counts = []
        for raw in report["outputs"]:
            record = lookup[raw["id"]]
            assert raw["source"] == record["source"] and raw["expected"] == record["target"]
            actual = counts_for_note(record, raw["generated"])
            assert actual == raw["counts"]
            counts.append(actual)
        measured = aggregate(counts)
        assert all(report["metrics"][key] == value for key, value in measured.items())
        assert report["metrics"]["valid_terminations"] == sum(raw["protocol_valid"] for raw in report["outputs"])
        rows.append(dict(split=split, **report["metrics"]))
        atomic_json(output / "summary.json", dict(identity, model_id=candidate["model_id"], rows=rows, status="complete" if len(rows) == 4 else "in progress", counts_recomputed=True, scope="four complete50-note partitions;200distinct notes shared with the primary eight-model comparison; selection frozen before these outputs"))
        print(f"Selected demo seed{candidate['seed']} {split}: {measured['exact']}/50 exact notes", flush=True)


if __name__ == "__main__":
    main()
