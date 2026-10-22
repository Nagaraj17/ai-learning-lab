"""Release the final single-corpus partitions only after saved author review.

The review freezes data/source/producer/protocol and the checkpoint rule. This
runner never trains a model or uses test outputs to choose its weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from assignment_evaluate import generated_metrics_v2
from config import ARCHITECTURES
from data import digest
from evaluate import token_metrics
from task_runtime import load_task_checkpoint, task_source_hash
from train import atomic_json


def run_final_partitions(root, review_path, device="cpu"):
    root, review_path = Path(root), Path(review_path)
    if not review_path.exists(): raise ValueError("Final partitions need a saved passed review before release")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    records = json.loads((root / "dataset/examples.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "results/protocol.json").read_text(encoding="utf-8"))
    identities = {"source_hash": task_source_hash(), "dataset_hash": digest(records), "protocol_hash": digest(protocol), "data_producer_source_hash": protocol["data_producer_source_hash"]}
    if review.get("status") != "passed" or not review.get("no_test_tuning") or any(review.get(key) != value for key, value in identities.items()):
        raise ValueError("Saved final review is missing or identities differ")
    if protocol["source_hash"] != identities["source_hash"] or protocol["dataset_hash"] != identities["dataset_hash"]:
        raise ValueError("Registered final source/data differs")
    if protocol.get("producer_files") and hashlib.sha256(b"".join(Path(name).read_bytes() for name in protocol["producer_files"])).hexdigest() != protocol["data_producer_source_hash"]:
        raise ValueError("Frozen data producer source differs")
    audit_path = root / "results/audit.json"
    if not audit_path.exists() or not json.loads(audit_path.read_text(encoding="utf-8")).get("engineering_audit_passed"):
        raise ValueError("Completed-model engineering audit is required before final release")
    summary = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
    if len(summary) != 8 or any(row["selected_step"] != protocol["steps"] for row in summary):
        raise ValueError("All eight frozen final-update runs must finish before release")
    output = root / "results/final_test"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    splits = sorted({row["split"] for row in records if row["split"].startswith("test_")})
    if len(splits) != 4: raise ValueError("Expected four final partitions")
    for architecture in ARCHITECTURES:
        path = root / f"checkpoints/{architecture}_seed42/latest.pt"
        checkpoint_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        model, tokenizer, checkpoint = load_task_checkpoint(path, device)
        if model.config.architecture != architecture or checkpoint["step"] != protocol["steps"] or checkpoint["dataset_hash"] != identities["dataset_hash"] or checkpoint["source_hash"] != identities["source_hash"]:
            raise ValueError("Final checkpoint identity differs")
        for split in splits:
            selected = [row for row in records if row["split"] == split]
            if len(selected) != 50: raise ValueError("Expected all50 notes in each final partition")
            target = output / f"{architecture}_{split}.json"
            expected = dict(identities, checkpoint_sha256=checkpoint_sha, checkpoint_step=checkpoint["step"], tokenizer_hash=tokenizer.fingerprint)
            if target.exists():
                result = json.loads(target.read_text(encoding="utf-8"))
                if any(result.get(key) != value for key, value in expected.items()): raise ValueError("Existing final outputs differ; preserve/review before replacing")
            else:
                result = generated_metrics_v2(model, selected, tokenizer)
                result.update(expected, architecture=architecture, seed=42, split=split, teacher_forced=token_metrics(model, selected, tokenizer), scope="complete frozen final partition; no tuning from these outcomes")
                atomic_json(target, result)
            if {row["id"] for row in result["outputs"]} != {row["id"] for row in selected} or len(result["outputs"]) != 50:
                raise ValueError("Saved final partition coverage differs")
            rows.append(dict(architecture=architecture, split=split, **result["metrics"]))
            atomic_json(output / "summary.json", rows)
            print(json.dumps({"architecture": architecture, "split": split, "exact": result["metrics"]["exact"], "examples": 50}), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/mitra_final")
    parser.add_argument("--review", default="reports/FINAL_PROTOCOL_REVIEW.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_final_partitions(args.root, args.review, args.device)


if __name__ == "__main__":
    main()
