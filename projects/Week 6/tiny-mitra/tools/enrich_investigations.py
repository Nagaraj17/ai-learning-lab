"""Add exact-query probabilities, intervention loss and category precision."""
import json
from pathlib import Path
import torch
from data import load_dataset
from diagnostics import category_precision, conditional_probe, masked_token_metrics
from train import atomic_json, load_checkpoint


def main():
    torch.set_num_threads(2)
    records = load_dataset()
    root = Path("artifacts/results/investigations")
    baseline = json.loads((root / "baseline.json").read_text(encoding="utf-8"))
    subset = [record for record in records if record["id"] in baseline["example_ids"]]
    model, tokenizer, metadata = load_checkpoint(baseline["checkpoint"])
    entries = []
    files = [root / "baseline.json"] + sorted(root.glob("layer*_head*.json")) + sorted(root.glob("pair_layer*.json"))
    for path in files:
        saved = json.loads(path.read_text(encoding="utf-8"))
        change = saved.get("intervention")
        gates = None
        if change:
            gates = {change["layer"]: [0 if head in change["disabled_heads"] else 1 for head in range(model.config.n_heads)]}
        entries.append({"experiment_id": saved["experiment_id"], "intervention": change, "teacher_forced": masked_token_metrics(model, subset, tokenizer, gates), "generated_metrics": saved["metrics"], "category_precision_recall": category_precision(records, saved["outputs"]), "raw_evidence": str(path)})
    atomic_json(root / "enriched_head_measurements.json", {"run_id": metadata["run_id"], "checkpoint_step": metadata["step"], "source_hash": metadata["source_hash"], "dataset_hash": metadata["dataset_hash"], "example_ids": baseline["example_ids"], "measurements": entries, "note": "teacher-forced loss and free-generation metrics are distinct; source checkpoint unchanged"})
    indexed = {entry["experiment_id"]: entry for entry in entries}
    pair_comparisons = []
    for entry in entries:
        intervention = entry["intervention"]
        if not intervention or len(intervention["disabled_heads"]) != 2:
            continue
        layer = intervention["layer"]
        singles = [indexed[f"head_{layer}_{head}"] for head in intervention["disabled_heads"]]
        measures = {}
        for metric in ("recall", "exact_match", "clinical_preservation"):
            reference = indexed["head_baseline"]["generated_metrics"][metric]
            single_deltas = [single["generated_metrics"][metric] - reference for single in singles]
            pair_delta = entry["generated_metrics"][metric] - reference
            measures[metric] = {"baseline": reference, "single_deltas": single_deltas, "pair_delta": pair_delta, "pair_minus_sum_of_single_deltas": pair_delta - sum(single_deltas)}
        pair_comparisons.append({"layer": layer, "heads": intervention["disabled_heads"], "measurements": measures})
    atomic_json(root / "pair_interactions.json", {"comparisons": pair_comparisons, "scope": "exploratory validation-selected pairs; metric floors/ceilings and tiny support limit interaction interpretation"})
    prefix = "PATIENT [NAME] DIAGNOSIS NSCLC BIOMARKER EGFR TREATMENT Drug"
    sources = [f"PATIENT John Lee DIAGNOSIS NSCLC BIOMARKER EGFR TREATMENT Drug{letter}" for letter in "AB"]
    models = []
    for architecture in ("embedding_only", "single_attention", "multi_attention", "transformer_2"):
        candidate, candidate_tokenizer, checkpoint = load_checkpoint(f"artifacts/checkpoints/{architecture}_seed42/best.pt")
        observations = [conditional_probe(candidate, candidate_tokenizer, source, prefix, expected) for source, expected in zip(sources, "AB")]
        if architecture == "embedding_only":
            assert observations[0]["candidates"] == observations[1]["candidates"], "Equal-position control failed"
        models.append({"architecture": architecture, "selected_step": checkpoint["step"], "correct_conditional_next_tokens": sum(row["chosen"] == row["expected"] for row in observations), "observations": observations})
    # MRN spelling changes should not change its target placeholder; this probe
    # investigates sensitivity, not a guarantee that a particular head handles MRNs.
    mrn_sources = [f"PATIENT John Lee MRN 7000000{digit} DIAGNOSIS NSCLC TREATMENT DrugA" for digit in "12"]
    mrn = [conditional_probe(model, tokenizer, source, "PATIENT [NAME] MRN ", "[MRN]") for source in mrn_sources]
    atomic_json(root / "conditional_probabilities.json", {"question": "Does one source-token change affect the exact same next-token query?", "drug_pair": models, "mrn_counterfactual": mrn, "scope": "authored probes with supplied prefixes; no semantic head label assigned"})
    print("Enriched baseline, eight single-head, two pair and token-controlled probability probes")


if __name__ == "__main__":
    main()
