"""Render final-corpus figures and descriptive training/repeat summaries.

Counts and curves come from saved runs. No test-based model selection, invented
timings or statistical significance claims are made by these figures.
"""
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from config import ARCHITECTURES
from train import atomic_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    root, reports = Path("artifacts/mitra_final"), Path("reports")
    rows = read(root / "results/summary.json")
    repeats = read(root / "results/repeats_summary.json")
    assert len(rows) == len(repeats) == 8
    assert read(root / "results/audit.json")["engineering_audit_passed"]
    assert read(root / "results/repeats_audit.json")["passed"]
    by_name = {row["architecture"]: row for row in rows}
    measured = {name: read(root / f"results/{name}_validation.json")["metrics"] for name in ARCHITECTURES}
    names = list(ARCHITECTURES)
    y = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(11, 6))
    for offset, key, label in ((-0.25, "exact_match", "Complete notes"), (0, "recall", "Correct identifier replacements"), (0.25, "clinical_preservation", "Preserved clinical associations")):
        axis.barh(y + offset, [measured[name][key] for name in names], height=0.23, label=label)
    axis.set(yticks=y, yticklabels=names, xlim=(0, 1.02), xlabel="Fraction correct within each metric's support", title="Final corpus: seed 42, fixed 4,000 updates; 24 validation diagnostics")
    axis.invert_yaxis()
    axis.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    figure.tight_layout()
    figure.savefig(reports / "final_architecture_quality.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    stability = []
    for name in names:
        history = read(root / f"checkpoints/{name}_seed42/history.json")
        steps = [row["step"] for row in history]
        losses = [row["validation"]["loss"] for row in history]
        norms = [row["gradient_norm"] for row in history]
        axes[0].plot(steps, losses, label=name)
        axes[1].plot(steps, norms, label=name)
        stability.append({"architecture": name, "parameters": by_name[name]["parameters"], "updates": 4000, "logged_gradient_observations": len(norms), "maximum_logged_gradient_norm_before_clip": max(norms), "logged_observations_above_clip_threshold1": sum(value > 1 for value in norms), "largest_positive_validation_loss_change_between_logged_updates": max([0] + [right - left for left, right in zip(losses, losses[1:])]), "durable_run_elapsed_seconds": history[-1]["elapsed_seconds"], "timing_limit": "local concurrent workers; not an isolated architecture speed benchmark"})
    axes[0].set(xlabel="Optimizer updates", ylabel="Output-token-weighted validation loss", title="Same tokenizer and 100 validation notes")
    axes[1].set(xlabel="Optimizer updates", ylabel="Logged norm before clipping (log scale)", title="Sampled training diagnostics; not every update", yscale="log")
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(reports / "final_training_curves.png", dpi=160)
    plt.close(figure)
    ranges = []
    for name in ("embedding_only", "single_attention", "multi_attention", "transformer_2"):
        selected = [by_name[name]] + [row for row in repeats if row["architecture"] == name]
        assert {row["seed"] for row in selected} == {42, 7, 123}
        entry = {"architecture": name, "seeds": [row["seed"] for row in selected]}
        for metric in ("diagnostic_exact", "diagnostic_recall", "diagnostic_clinical_preservation", "acceptance_correct_notes"):
            values = [row[metric] for row in selected]
            entry[metric] = {"mean": statistics.mean(values), "minimum": min(values), "maximum": max(values), "values": values}
        ranges.append(entry)
    head = read(root / "results/investigations/enriched_head_measurements.json")["measurements"]
    baseline, singles = head[0], [row for row in head if row["experiment_id"].startswith("head_") and row["experiment_id"] != "head_baseline"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = [row["experiment_id"].replace("head_", "L").replace("_", "H") for row in singles]
    for axis, key, title in ((axes[0], "recall", "Identifier recall change; negative means harm"), (axes[1], "loss", "Conditional loss change; positive means harm")):
        def value(row):
            return row["teacher_forced"]["loss"] if key == "loss" else row["generated_metrics"][key]
        axis.bar(labels, [value(row) - value(baseline) for row in singles])
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set(title=title, xlabel="Disabled head; seed 42 / update 4,000")
    figure.suptitle("Same eight validation notes; head dependence is metric-specific")
    figure.tight_layout()
    figure.savefig(reports / "final_head_effects.png", dpi=160)
    plt.close(figure)
    atomic_json(root / "results/descriptive_comparison.json", {"training_diagnostics": stability, "training_seed_ranges": ranges, "limits": "three seeds vary initialization and minibatch order on shared development examples; descriptive ranges, not significance or independent datasets; component variants have one primary seed; fewer parameters and equal updates are not equal compute"})
    print("Rendered three final-corpus figures and measured seed/stability summaries")


if __name__ == "__main__":
    main()
