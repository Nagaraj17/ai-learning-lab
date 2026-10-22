"""Registered representation study: same assignment data, fixed task-word tokens.

All architectures use identical tokenizer/data/update settings within this
series. Character-study weights/results stay intact and are not relabelled.
"""
import argparse
import json
from pathlib import Path
import torch
from assignment_data import create_dataset, instructor_examples, validate
from assignment_evaluate import generated_metrics_v2
from config import ARCHITECTURES, ModelConfig
from data import training_item
from experiments import diagnostic_subset
from task_runtime import load_task_checkpoint, task_source_hash, train_task
from task_tokenizer import TaskTokenizer
from train import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/task_tokens")
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=["transformer_2"] + [name for name in ARCHITECTURES if name != "transformer_2"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1: parser.error("Total updates must be positive")
    torch.set_num_threads(2)
    root, tokenizer, records = Path(args.root), TaskTokenizer(), create_dataset()
    manifest = validate(records)
    character_maximum = manifest["maximum_full_tokens"]
    lengths = [len(training_item(row, tokenizer, 256)[0]) + 1 for row in records]
    manifest.update(maximum_character_full_tokens=character_maximum, maximum_full_tokens=max(lengths), tokenizer_hash=tokenizer.fingerprint, vocabulary_size=len(tokenizer), tokenization="fixed task pieces with character fallback; original source strings and split identities unchanged")
    protocol = {"series": "fixed_task_words_character_fallback", "dataset_hash": manifest["dataset_hash"], "source_hash": task_source_hash(), "tokenizer": tokenizer.metadata(), "architectures": args.architectures, "steps": args.steps, "seed": args.seed, "batch_size": 16, "lr": 0.0007, "context": 256, "width": 64, "heads": 4, "ffn": 192, "selection": "fixed final update for every architecture; minimum-loss checkpoints retained separately", "purpose": "test whether shorter task-word representation helps the same assignment grammar; no pretrained weights or inference replacement rules", "evaluation": "24 development diagnostics and 11 authored prompts; final partitions protected", "cross_tokenizer_limit": "loss/perplexity per token are not comparable between tokenizations; vocabulary/sequence lengths and parameter counts differ; older character benchmarks use minimum-loss selection, so matched final-update comparisons are required for a representation claim"}
    root.mkdir(parents=True, exist_ok=True)
    saved_protocol = root / "results/protocol.json"
    if saved_protocol.exists() and json.loads(saved_protocol.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Registered task-token protocol differs; use a new reviewed series")
    for path, value in (("results/protocol.json", protocol), ("dataset/manifest.json", manifest), ("dataset/examples.json", records)):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        atomic_json(root / path, value)
    validation = diagnostic_subset([row for row in records if row["split"] == "validation"], 24)
    summaries = []
    for architecture in args.architectures:
        directory = root / f"checkpoints/{architecture}_seed{args.seed}"
        config = ModelConfig(len(tokenizer), architecture=architecture, context=256)
        model, tokenizer, checkpoint = train_task(config, records, directory, steps=args.steps, seed=args.seed, device=args.device, resume=(directory / "latest.pt").exists())
        # A fixed-update comparison keeps optimization exposure common. The
        # best-loss checkpoint is still saved, but never silently substituted.
        model, tokenizer, checkpoint = load_task_checkpoint(directory / "latest.pt", args.device)
        reports = {}
        for name, selected in (("validation", validation), ("acceptance", instructor_examples())):
            report = generated_metrics_v2(model, selected, tokenizer)
            report.update(series="task_tokens", architecture=architecture, selected_step=checkpoint["step"], dataset_hash=checkpoint["dataset_hash"], source_hash=checkpoint["source_hash"], tokenizer_hash=tokenizer.fingerprint, scope="development evidence, not final test")
            atomic_json(root / f"results/{architecture}_{name}.json", report)
            reports[name] = report["metrics"]
        summaries.append({"architecture": architecture, "seed": args.seed, "selected_step": checkpoint["step"], "updates": args.steps, "selection": "fixed final update", "parameters": model.parameter_count, "validation_loss": checkpoint["history"][-1]["validation"]["loss"], "validation_token_accuracy": checkpoint["history"][-1]["validation"]["next_token_accuracy"], "diagnostic_exact": reports["validation"]["exact_match"], "diagnostic_recall": reports["validation"]["recall"], "diagnostic_clinical_preservation": reports["validation"]["clinical_preservation"], "acceptance_correct_notes": reports["acceptance"]["exact"], "acceptance_notes": reports["acceptance"]["examples"]})
        atomic_json(root / "results/summary.json", summaries)
        print(json.dumps(summaries[-1]), flush=True)


if __name__ == "__main__":
    main()
