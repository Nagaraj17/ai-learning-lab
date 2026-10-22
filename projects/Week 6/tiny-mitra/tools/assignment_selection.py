"""Compare the fixed best-loss rule with the full model's last update.

Both are evaluated on the same development IDs. This exploratory diagnostic
does not replace the registered best-loss architecture benchmark or use final
test outcomes to select a checkpoint.
"""
import json
from pathlib import Path
import torch
from assignment_data import create_dataset, instructor_examples
from assignment_evaluate import generated_metrics_v2
from experiments import diagnostic_subset
from train import atomic_json, load_checkpoint


def main():
    torch.set_num_threads(2)
    root = Path("artifacts/assignment_v2")
    manifest = json.loads((root / "checkpoints/transformer_2_seed42/manifest.json").read_text(encoding="utf-8"))
    if manifest["completed_steps"] != 2000: raise ValueError("Wait for the full model's registered budget")
    records = create_dataset()
    validation = diagnostic_subset([row for row in records if row["split"] == "validation"], 24)
    model, tokenizer, checkpoint = load_checkpoint(root / "checkpoints/transformer_2_seed42/latest.pt")
    if checkpoint["step"] != 2000: raise ValueError("Latest checkpoint is not the completed budget")
    rows = []
    for kind, selected in (("validation", validation), ("acceptance", instructor_examples())):
        latest = generated_metrics_v2(model, selected, tokenizer)
        latest.update(version="assignment_v2", selected_step=checkpoint["step"], dataset_hash=checkpoint["dataset_hash"], source_hash=checkpoint["source_hash"], scope="exploratory latest-update development diagnostic; benchmark selection unchanged")
        atomic_json(root / f"results/transformer_2_latest_{kind}.json", latest)
        best = json.loads((root / f"results/transformer_2_{kind}.json").read_text(encoding="utf-8"))
        assert best["metrics"]["example_ids"] == latest["metrics"]["example_ids"]
        rows.append({"development_subset": kind, "best_update": best["selected_step"], "latest_update": checkpoint["step"], "best_metrics": best["metrics"], "latest_metrics": latest["metrics"]})
    atomic_json(root / "results/checkpoint_selection.json", {"scope": "matched development IDs; selection-rule investigation, no benchmark replacement or final-test tuning", "comparisons": rows})
    print(json.dumps([{key: row[key] for key in ("development_subset", "best_update", "latest_update")} | {"best_exact": row["best_metrics"]["exact"], "latest_exact": row["latest_metrics"]["exact"], "notes": row["latest_metrics"]["examples"]} for row in rows], indent=2))


if __name__ == "__main__":
    main()
