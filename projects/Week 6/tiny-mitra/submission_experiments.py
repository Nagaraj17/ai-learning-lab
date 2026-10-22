"""Final comparison: one corpus of1,000 examples, random weights, eight models.

This is the main assignment runner. Earlier representation/continued-training
trials remain separate evidence and supply no weights to these models.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from assignment_data import instructor_examples
from assignment_evaluate import generated_metrics_v2
from broad_data import save_broad_dataset
from config import ARCHITECTURES, ModelConfig
from experiments import diagnostic_subset
from task_runtime import load_task_checkpoint, task_source_hash, train_task
from train import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/mitra_final")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.root)
    records, manifest = save_broad_dataset(root)
    producer_files = ["broad_data.py", "submission_experiments.py"]
    producer_hash = hashlib.sha256(b"".join(Path(name).read_bytes() for name in producer_files)).hexdigest()
    order = ["transformer_2"] + [name for name in ARCHITECTURES if name != "transformer_2"]
    protocol = {"series": "final single-corpus comparison", "dataset_hash": manifest["dataset_hash"], "source_hash": task_source_hash(), "data_producer_source_hash": producer_hash, "producer_files": producer_files, "initialization": "random weights, seed 42; no weights from earlier experiments", "architectures": order, "steps": 4000, "batch_size": 16, "lr": 0.0007, "seed": 42, "width": 64, "ffn": 192, "heads": 4, "context": 256, "device": args.device, "selection": "fixed final update for all architectures; minimum-loss checkpoints saved separately", "data": "one saved corpus of 1,000 examples: 700 train, 100 validation and four final partitions of 50 notes; authored development sources excluded; final test names exclude development names", "evaluation": "24 declared validation diagnostics and 11 authored development prompts; final partitions released only after review", "limits": "synthetic grammar, single seed initially; exact-surface leakage is not complete clinical safety; different tokenizers/selection histories must not be mixed"}
    (root / "results").mkdir(parents=True, exist_ok=True)
    path = root / "results/protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Final registered protocol differs; preserve it and use a reviewed new series")
    atomic_json(path, protocol)
    snapshot = root / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in producer_files:
        content, destination = Path(name).read_bytes(), snapshot / name
        if destination.exists() and destination.read_bytes() != content: raise ValueError("Producer source snapshot changed")
        destination.write_bytes(content)
    validation = diagnostic_subset([row for row in records if row["split"] == "validation"], 24)
    summaries = []
    for architecture in order:
        directory = root / f"checkpoints/{architecture}_seed42"
        config = ModelConfig(manifest["vocabulary_size"], architecture=architecture, context=256)
        train_task(config, records, directory, steps=4000, device=args.device, resume=(directory / "latest.pt").exists())
        model, tokenizer, checkpoint = load_task_checkpoint(directory / "latest.pt", args.device)
        metrics = {}
        for name, selected in (("validation", validation), ("acceptance", instructor_examples())):
            report = generated_metrics_v2(model, selected, tokenizer)
            report.update(series="final single-corpus scratch comparison", architecture=architecture, selected_step=4000, dataset_hash=manifest["dataset_hash"], source_hash=checkpoint["source_hash"], tokenizer_hash=tokenizer.fingerprint, scope="development evidence; no final-partition tuning")
            atomic_json(root / f"results/{architecture}_{name}.json", report)
            metrics[name] = report["metrics"]
        last = checkpoint["history"][-1]["validation"]
        summaries.append({"architecture": architecture, "seed": 42, "selected_step": 4000, "updates": 4000, "selection": "fixed final update from random weights", "parameters": model.parameter_count, "validation_loss": last["loss"], "validation_token_accuracy": last["next_token_accuracy"], "diagnostic_exact": metrics["validation"]["exact_match"], "diagnostic_recall": metrics["validation"]["recall"], "diagnostic_clinical_preservation": metrics["validation"]["clinical_preservation"], "acceptance_correct_notes": metrics["acceptance"]["exact"], "acceptance_notes": metrics["acceptance"]["examples"]})
        atomic_json(root / "results/summary.json", summaries)
        print(json.dumps(summaries[-1]), flush=True)


if __name__ == "__main__":
    main()
