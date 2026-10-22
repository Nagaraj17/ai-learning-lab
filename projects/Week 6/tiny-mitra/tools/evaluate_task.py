"""Development metrics for task-token cloud runs, using their final updates."""
import argparse
import json
from pathlib import Path
import torch
from assignment_data import create_dataset, instructor_examples, validate
from assignment_evaluate import generated_metrics_v2
from config import ARCHITECTURES
from experiments import diagnostic_subset
from task_runtime import load_task_checkpoint
from train import atomic_json
from data import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/task_tokens_cloud")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", default="artifacts/mitra_final/dataset/examples.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.root)
    records = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    from broad_data import validate_broad
    manifest = validate_broad(records) if records[0].get("dataset_version", "").startswith("broad_coverage_") else validate(records)
    (root / "dataset").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    atomic_json(root / "dataset/manifest.json", manifest)
    atomic_json(root / "dataset/examples.json", records)
    validation = diagnostic_subset([row for row in records if row["split"] == "validation"], 24)
    summaries = []
    for architecture in ARCHITECTURES:
        directory = root / f"checkpoints/{architecture}_seed42"
        model, tokenizer, checkpoint = load_task_checkpoint(directory / "latest.pt", args.device)
        if checkpoint["dataset_hash"] != digest(records) or model.config.architecture != architecture:
            raise ValueError("Evaluation dataset/architecture differs from the trained checkpoint")
        reports = {}
        for name, selected in (("validation", validation), ("acceptance", instructor_examples())):
            report = generated_metrics_v2(model, selected, tokenizer)
            report.update(series="task_tokens", architecture=architecture, selected_step=checkpoint["step"], dataset_hash=checkpoint["dataset_hash"], source_hash=checkpoint["source_hash"], tokenizer_hash=tokenizer.fingerprint, scope="development evidence; not final test")
            atomic_json(root / f"results/{architecture}_{name}.json", report)
            reports[name] = report["metrics"]
        last = checkpoint["history"][-1]["validation"]
        summaries.append({"architecture": architecture, "seed": 42, "selected_step": checkpoint["step"], "updates": checkpoint["step"], "selection": "fixed final update", "parameters": model.parameter_count, "validation_loss": last["loss"], "validation_token_accuracy": last["next_token_accuracy"], "diagnostic_exact": reports["validation"]["exact_match"], "diagnostic_recall": reports["validation"]["recall"], "diagnostic_clinical_preservation": reports["validation"]["clinical_preservation"], "acceptance_correct_notes": reports["acceptance"]["exact"], "acceptance_notes": reports["acceptance"]["examples"]})
    atomic_json(root / "results/summary.json", summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
