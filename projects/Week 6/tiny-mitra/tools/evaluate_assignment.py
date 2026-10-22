"""Evaluate completed assignment-format checkpoints on development prompts.

Works for a separate cloud output root too. No final-test partition is released.
"""
import argparse
import json
from pathlib import Path
import torch
from assignment_data import instructor_examples, save_dataset
from assignment_evaluate import generated_metrics_v2
from config import ARCHITECTURES
from experiments import diagnostic_subset
from train import atomic_json, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/assignment_v2_cloud")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.root)
    records, data_manifest = save_dataset()
    # Each series carries the same synthetic data identity next to its weights.
    atomic_json(root / "dataset/manifest.json", data_manifest)
    atomic_json(root / "dataset/examples.json", records)
    validation = diagnostic_subset([row for row in records if row["split"] == "validation"], 24)
    summaries = []
    for architecture in args.architectures:
        directory = root / f"checkpoints/{architecture}_seed42"
        model, tokenizer, checkpoint = load_checkpoint(directory / "best.pt", args.device)
        if checkpoint["dataset_hash"] != data_manifest["dataset_hash"] or model.config.architecture != architecture:
            raise ValueError("Checkpoint dataset/architecture differs")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        diagnostic = generated_metrics_v2(model, validation, tokenizer)
        acceptance = generated_metrics_v2(model, instructor_examples(), tokenizer)
        for name, report in (("validation", diagnostic), ("acceptance", acceptance)):
            report.update(version="assignment_v2", architecture=architecture, selected_step=checkpoint["step"], source_hash=checkpoint["source_hash"], dataset_hash=data_manifest["dataset_hash"], scope="validation or authored development prompts; not final test")
            atomic_json(root / f"results/{architecture}_{name}.json", report)
        summaries.append({"architecture": architecture, "seed": 42, "selected_step": checkpoint["step"], "updates": manifest["completed_steps"], "parameters": model.parameter_count, "validation_loss": manifest["validation"]["loss"], "validation_token_accuracy": manifest["validation"]["next_token_accuracy"], "diagnostic_exact": diagnostic["metrics"]["exact_match"], "diagnostic_recall": diagnostic["metrics"]["recall"], "diagnostic_clinical_preservation": diagnostic["metrics"]["clinical_preservation"], "acceptance_correct_notes": acceptance["metrics"]["exact"], "acceptance_notes": len(acceptance["outputs"])})
    atomic_json(root / "results/summary.json", summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
