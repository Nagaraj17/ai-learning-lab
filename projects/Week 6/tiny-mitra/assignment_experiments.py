"""Separate registered comparison covering the instructor's sentence forms.

The first eight-run study remains intact. This series uses a new dataset,
context/budget and evaluator; do not combine its numbers with version 1.
"""
import hashlib
import json
from pathlib import Path
import torch
from assignment_data import instructor_examples, save_dataset
from assignment_evaluate import generated_metrics_v2
from config import ARCHITECTURES, ModelConfig
from data import digest
from evaluate import token_metrics
from experiments import diagnostic_subset
from tokenizer import CharacterTokenizer
from train import atomic_json, source_hash, train_run


def main():
    torch.set_num_threads(2)
    root = Path("artifacts/assignment_v2")
    (root / "results").mkdir(parents=True, exist_ok=True)
    records, manifest = save_dataset()
    protocol = {"version": "assignment_v2", "reason": "Instructor-format coverage gap found in source review: add VISITED/ON, LIVES, DOB, MRN orderings and all clinical preservation terms. Version-1 final outcomes do not tune this series.", "dataset_hash": digest(records), "core_source_hash": source_hash(), "assignment_source_hash": hashlib.sha256(b"".join(Path(name).read_bytes() for name in ("assignment_data.py", "assignment_evaluate.py", "assignment_experiments.py"))).hexdigest(), "architectures": list(ARCHITECTURES), "context": 256, "width": 64, "ffn": 192, "heads": 4, "seed": 42, "steps": 2000, "lr": 0.0007, "batch_size": 16, "selection": "minimum token-weighted validation cross entropy; latest reported separately", "evaluation": "11 authored instructor/clinical acceptance prompts plus 24 validation diagnostics; final partitions remain untouched until review", "primary_generated_criteria": ["exact note match", "correct typed replacements", "clinical field/value preservation", "exact-surface leakage"], "threshold": "descriptive comparison; no significance or universal-superiority claim"}
    path = root / "results/protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Assignment-series protocol changed; preserve/review before restarting")
    atomic_json(path, protocol)
    validation = [row for row in records if row["split"] == "validation"]
    subset = diagnostic_subset(validation, 24)
    summaries = []
    # Prioritize the full model for a usable assignment demonstration; remaining
    # models use the same fixed settings and selection rule.
    order = ["transformer_2", "embedding_only", "single_attention", "multi_attention", "transformer_1", "no_ffn", "no_layernorm", "no_residual"]
    for architecture in order:
        directory = root / f"checkpoints/{architecture}_seed42"
        config = ModelConfig(len(CharacterTokenizer()), architecture=architecture, context=256)
        model, tokenizer = train_run(config, records, directory, steps=2000, batch_size=16, lr=0.0007, seed=42, validate_every=100, resume=(directory / "latest.pt").exists())
        saved = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        diagnostics = generated_metrics_v2(model, subset, tokenizer)
        acceptance = generated_metrics_v2(model, instructor_examples(), tokenizer)
        for name, value in (("validation", diagnostics), ("acceptance", acceptance)):
            value.update(version="assignment_v2", architecture=architecture, selected_step=saved["selected_step"], source_hash=saved["source_hash"], dataset_hash=manifest["dataset_hash"], scope="validation or authored acceptance, not final test")
            atomic_json(root / f"results/{architecture}_{name}.json", value)
        summaries.append({"architecture": architecture, "seed": 42, "selected_step": saved["selected_step"], "updates": 2000, "parameters": model.parameter_count, "validation_loss": saved["validation"]["loss"], "validation_token_accuracy": saved["validation"]["next_token_accuracy"], "diagnostic_exact": diagnostics["metrics"]["exact_match"], "diagnostic_recall": diagnostics["metrics"]["recall"], "diagnostic_clinical_preservation": diagnostics["metrics"]["clinical_preservation"], "acceptance_exact": acceptance["metrics"]["exact_match"], "acceptance_correct_notes": acceptance["metrics"]["exact"], "acceptance_notes": len(acceptance["outputs"])})
        atomic_json(root / "results/summary.json", summaries)
        print(json.dumps(summaries[-1]), flush=True)


if __name__ == "__main__":
    main()
