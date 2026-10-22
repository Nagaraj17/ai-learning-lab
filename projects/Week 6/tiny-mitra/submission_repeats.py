"""Registered final-corpus training-seed repeats; primary seed42 stays unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from assignment_data import instructor_examples
from assignment_evaluate import generated_metrics_v2
from config import ModelConfig
from data import digest
from experiments import diagnostic_subset
from task_runtime import load_task_checkpoint, task_source_hash, train_task
from task_tokenizer import TaskTokenizer
from train import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/mitra_final")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.root)
    records = json.loads((root / "dataset/examples.json").read_text(encoding="utf-8"))
    primary = json.loads((root / "results/protocol.json").read_text(encoding="utf-8"))
    if primary["dataset_hash"] != digest(records) or primary["source_hash"] != task_source_hash(): raise ValueError("Final source/data identity differs")
    order, seeds = ["transformer_2", "embedding_only", "single_attention", "multi_attention"], [7, 123]
    protocol = {"dataset_hash": digest(records), "source_hash": task_source_hash(), "runner_source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "architectures": order, "seeds": seeds, "steps": 4000, "lr": 0.0007, "batch_size": 16, "context": 256, "selection": "fixed final update", "seed_changes": "weight initialization and sampled minibatch order", "scope": "same24 validation diagnostics and11 authored development prompts; no final-test seed selection", "primary_unchanged": "seed42 remains the registered primary architecture comparison; any demo checkpoint must be separately identified and selected from development evidence"}
    path = root / "results/repeats_protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol: raise ValueError("Final repeat protocol differs")
    atomic_json(path, protocol)
    validation = diagnostic_subset([row for row in records if row["split"] == "validation"], 24)
    summaries = []
    for architecture in order:
        for seed in seeds:
            directory = root / f"checkpoints/{architecture}_seed{seed}"
            config = ModelConfig(len(TaskTokenizer()), architecture=architecture, context=256)
            train_task(config, records, directory, steps=4000, seed=seed, device=args.device, resume=(directory / "latest.pt").exists())
            model, tokenizer, checkpoint = load_task_checkpoint(directory / "latest.pt", args.device)
            metrics = {}
            for name, selected in (("validation", validation), ("acceptance", instructor_examples())):
                result = generated_metrics_v2(model, selected, tokenizer)
                result.update(series="final training-seed repeats", architecture=architecture, seed=seed, selected_step=4000, dataset_hash=digest(records), source_hash=checkpoint["source_hash"], tokenizer_hash=tokenizer.fingerprint, scope="development diagnostics, not final-test performance")
                atomic_json(root / f"results/{architecture}_seed{seed}_{name}.json", result)
                metrics[name] = result["metrics"]
            last = checkpoint["history"][-1]["validation"]
            summaries.append({"architecture": architecture, "seed": seed, "selected_step": 4000, "updates": 4000, "parameters": model.parameter_count, "validation_loss": last["loss"], "validation_token_accuracy": last["next_token_accuracy"], "diagnostic_exact": metrics["validation"]["exact_match"], "diagnostic_recall": metrics["validation"]["recall"], "diagnostic_clinical_preservation": metrics["validation"]["clinical_preservation"], "acceptance_correct_notes": metrics["acceptance"]["exact"], "acceptance_notes": metrics["acceptance"]["examples"]})
            atomic_json(root / "results/repeats_summary.json", summaries)
            print(json.dumps(summaries[-1]), flush=True)


if __name__ == "__main__":
    main()
