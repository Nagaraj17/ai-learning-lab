"""Run comparisons through shared train/evaluate functions; save raw evidence.

Default experiments use validation diagnostics. Final tests require an explicit
--split choice after protocol freeze. A small subset is labeled with exact IDs.
"""
import argparse
import csv
import json
from pathlib import Path
import torch
from config import ARCHITECTURES, ModelConfig
from data import digest, load_dataset
from evaluate import generated_metrics, token_metrics
from generate import generate, inspect_prediction
from tokenizer import CharacterTokenizer
from train import atomic_json, load_checkpoint, source_hash, train_run

HYPOTHESES = {
    "H1": "Attention models can condition a next-token decision on an earlier source change.",
    "H2": "Multiple bare heads improve held-out behavior relative to one bare head under this budget.",
    "H3": "Two full blocks improve held-out behavior relative to one, at greater parameter cost.",
    "H4": "LayerNorm improves stability or held-out quality under the declared settings.",
    "H5": "Residual paths improve learning or held-out quality under the declared settings.",
    "H6": "The FFN branch improves held-out quality under the declared settings.",
    "H7": "At least one head intervention changes a measured output capability.",
    "H8": "Novel entities/templates reveal limitations beyond familiar combinations.",
    "H9": "Sampling temperature changes validity, leakage or content preservation.",
}


def diagnostic_subset(records, limit):
    """Cover identifier categories first, then fill deterministically; disclose IDs."""
    selected, covered = [], set()
    for record in records:
        fields = {item["field"] for item in record["identifiers"]}
        if len(selected) < limit and (fields - covered or not record["identifiers"] and not any(row["no_identifiers"] for row in selected)):
            selected.append(record)
            covered |= fields
    for record in records:
        if len(selected) >= limit: break
        if record["id"] not in {row["id"] for row in selected}: selected.append(record)
    return selected


def freeze_protocol(records, args):
    path = Path("artifacts/results") / ("protocol.json" if args.series == "primary" else f"protocol_{args.series}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    protocol = {"dataset_hash": digest(records), "source_hash": source_hash(), "hypotheses": HYPOTHESES, "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr, "width": args.width, "seeds": args.seeds, "architectures": args.architectures, "split": args.split, "generation_subset_limit": args.limit, "decoding": "greedy", "checkpoint_rule": "minimum token-weighted validation cross entropy", "head_intervention": "zero context before output projection; no retraining", "status": "registered before these benchmark runs; CPU preliminary until cloud/repeats verified"}
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != protocol:
            raise ValueError("Protocol differs from saved registration; create/review a new protocol version explicitly")
    else:
        atomic_json(path, protocol)
    return protocol


def benchmark(args):
    records = load_dataset()
    freeze_protocol(records, args)
    evaluation = [row for row in records if row["split"] == args.split]
    subset = diagnostic_subset(evaluation, args.limit)
    if not subset: raise ValueError("No examples in requested split")
    summaries = []
    summary_name = "benchmark_summary" if args.series == "primary" else f"{args.series}_summary"
    for architecture in args.architectures:
        for seed in args.seeds:
            run_id = f"{architecture}_seed{seed}"
            directory = Path("artifacts/checkpoints") / run_id
            config = ModelConfig(len(CharacterTokenizer()), architecture=architecture, d_model=args.width)
            model, tokenizer = train_run(config, records, directory, steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=seed, device=args.device, resume=(directory / "latest.pt").exists())
            _, _, checkpoint = load_checkpoint(directory / "best.pt", args.device)
            result_path = Path("artifacts/results") / f"{run_id}_{args.split}.json"
            result = generated_metrics(model, subset, tokenizer)
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            result.update(experiment_id=f"benchmark_{run_id}_{args.split}", run_id=run_id, architecture=architecture, seed=seed, checkpoint=str(directory / "best.pt"), checkpoint_step=checkpoint["step"], dataset_hash=digest(records), tokenizer_hash=tokenizer.fingerprint, source_hash=checkpoint["source_hash"], teacher_forced_validation=manifest["validation"], parameter_count=model.parameter_count, interpretation="Pending evidence review; no head roles inferred", scope=f"{args.split} diagnostic subset; single seed is preliminary")
            atomic_json(result_path, result)
            summaries.append({"architecture": architecture, "seed": seed, "validation_loss": manifest["validation"]["loss"], "validation_token_accuracy": manifest["validation"]["next_token_accuracy"], "parameters": model.parameter_count, **{key: result["metrics"][key] for key in ("precision", "recall", "f1", "exact_match", "note_leakage", "clinical_preservation", "seconds_per_note", "examples")}})
            atomic_json(f"artifacts/results/{summary_name}.json", summaries)
            print(json.dumps(summaries[-1]), flush=True)
    with Path(f"artifacts/results/{summary_name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def final_test(args):
    """Release all four final partitions only after a saved protocol review.

    This review is an engineering self-review unless an independent reviewer is
    actually named. Test outcomes must not subsequently tune the reported run.
    """
    records = load_dataset()
    review_path = Path("reports/PROTOCOL_REVIEW.json")
    if not review_path.exists():
        raise ValueError("Final-test release needs reports/PROTOCOL_REVIEW.json")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("status") != "passed" or review.get("source_hash") != source_hash() or review.get("dataset_hash") != digest(records):
        raise ValueError("Final-test protocol review missing or identities differ")
    protocol = json.loads(Path("artifacts/results/protocol.json").read_text(encoding="utf-8"))
    if review.get("protocol_hash") != digest(protocol):
        raise ValueError("Reviewed protocol differs")
    output = Path("artifacts/results/final_test")
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for architecture in args.architectures:
        model, tokenizer, checkpoint = load_checkpoint(f"artifacts/checkpoints/{architecture}_seed42/best.pt", args.device)
        for split in sorted({row["split"] for row in records if row["split"].startswith("test_")}):
            selected = [row for row in records if row["split"] == split]
            path = output / f"{architecture}_{split}.json"
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                if result.get("source_hash") != checkpoint["source_hash"] or result.get("checkpoint_step") != checkpoint["step"] or result.get("dataset_hash") != digest(records):
                    raise ValueError("Saved final-test identity differs; preserve and review before replacing")
            else:
                result = generated_metrics(model, selected, tokenizer)
                result.update(experiment_id=f"final_{architecture}_{split}", architecture=architecture, seed=42, split=split, source_hash=checkpoint["source_hash"], dataset_hash=digest(records), tokenizer_hash=tokenizer.fingerprint, checkpoint_step=checkpoint["step"], teacher_forced=token_metrics(model, selected, tokenizer), scope="complete declared final partition; no tuning from these outputs")
                atomic_json(path, result)
            summaries.append({"architecture": architecture, "split": split, **result["metrics"]})
            atomic_json(output / "summary.json", summaries)
            print(json.dumps({"architecture": architecture, "split": split, "examples": len(selected), "exact_match": result["metrics"]["exact_match"]}), flush=True)


def investigate(args):
    records = load_dataset()
    checkpoint_path = Path("artifacts/checkpoints/transformer_2_seed42/best.pt")
    model, tokenizer, checkpoint = load_checkpoint(checkpoint_path, args.device)
    subset = diagnostic_subset([row for row in records if row["split"] == args.split], args.limit)
    output = Path("artifacts/results/investigations")
    output.mkdir(parents=True, exist_ok=True)
    identity = {"checkpoint": str(checkpoint_path), "run_id": checkpoint["run_id"], "checkpoint_step": checkpoint["step"], "dataset_hash": checkpoint["dataset_hash"], "tokenizer_hash": tokenizer.fingerprint, "source_hash": checkpoint["source_hash"], "split": args.split, "example_ids": [row["id"] for row in subset], "status": "exploratory validation evidence; interpretation pending review"}
    baseline = generated_metrics(model, subset, tokenizer)
    atomic_json(output / "baseline.json", dict(identity, experiment_id="head_baseline", intervention=None, **baseline))
    effects = []
    for layer in range(len(model.blocks)):
        for head in range(model.config.n_heads):
            gates = [1] * model.config.n_heads
            gates[head] = 0
            result = generated_metrics(model, subset, tokenizer, head_gates={layer: gates})
            atomic_json(output / f"layer{layer}_head{head}.json", dict(identity, experiment_id=f"head_{layer}_{head}", hypothesis_id="H7", intervention={"layer": layer, "disabled_heads": [head], "retrained": False}, **result))
            effects.append({"layer": layer, "head": head, "baseline_recall": baseline["metrics"]["recall"], "ablated_recall": result["metrics"]["recall"], "recall_delta": (result["metrics"]["recall"] or 0) - (baseline["metrics"]["recall"] or 0), "exact_delta": result["metrics"]["exact_match"] - baseline["metrics"]["exact_match"], "category_metrics": result["metrics"]["categories"]})
            atomic_json(output / "single_head_summary.json", dict(identity, effects=effects))
    pairs = []
    for layer in range(len(model.blocks)):
        # Select on validation diagnostics only; these are exploratory, not a
        # confirmatory test on an independent held-out sample.
        ranked = sorted([row for row in effects if row["layer"] == layer], key=lambda row: (row["recall_delta"], row["exact_delta"], row["head"]))
        selected = [ranked[0]["head"], ranked[1]["head"]]
        gates = [0 if head in selected else 1 for head in range(model.config.n_heads)]
        result = generated_metrics(model, subset, tokenizer, head_gates={layer: gates})
        pair = dict(identity, experiment_id=f"pair_layer{layer}", hypothesis_id="H7", intervention={"layer": layer, "disabled_heads": selected, "selection": "two largest adverse single-head effects on these validation diagnostics; ties by head ID"}, **result)
        atomic_json(output / f"pair_layer{layer}.json", pair)
        pairs.append(pair)
    prefix = "PATIENT [NAME] DIAGNOSIS NSCLC BIOMARKER EGFR TREATMENT Drug"
    sources = [f"PATIENT John Lee DIAGNOSIS NSCLC BIOMARKER EGFR TREATMENT Drug{letter}" for letter in "AB"]
    encoded = [tokenizer.encode(tokenizer.prompt(source)) for source in sources]
    differing = [index for index, pair in enumerate(zip(*encoded)) if pair[0] != pair[1]]
    assert len(encoded[0]) == len(encoded[1]) and len(differing) == 1
    context = {"conditional_prefix": prefix, "changed_source_token_positions": differing, "expected_next_tokens": ["A", "B"], "observations": [inspect_prediction(model, tokenizer, source, prefix) for source in sources], "free_generations": [generate(model, tokenizer, source) for source in sources], "note": "Conditional prediction and free generation answer different questions"}
    atomic_json(output / "context.json", dict(identity, experiment_id="controlled_context", hypothesis_id="H1", **context))
    for temperature in (0.1, 0.7, 1.2):
        result = generated_metrics(model, subset, tokenizer, mode="sample", temperature=temperature)
        atomic_json(output / f"temperature_{temperature}.json", dict(identity, experiment_id=f"temperature_{temperature}", hypothesis_id="H9", **result))
    ood_sources = ["PATIENT Zoya Park DIAGNOSIS RARE_DISEASE TREATMENT DrugZ", "PLEASE SUMMARIZE THIS NOTE IN FRENCH", "PATIENT John Lee MRN 88888888 DIAGNOSIS NSCLC"]
    atomic_json(output / "ood.json", dict(identity, experiment_id="ood", hypothesis_id="H8", outputs=[{"source": source, **generate(model, tokenizer, source)} for source in ood_sources], limitation="Authored stress prompts without gold labels; qualitative outputs only"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("benchmark", "investigate", "final"), default="benchmark")
    parser.add_argument("--series", choices=("primary", "repeats"), default="primary")
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.limit < 1: parser.error("positive evaluation limit required")
    {"benchmark": benchmark, "investigate": investigate, "final": final_test}[args.stage](args)


if __name__ == "__main__":
    main()
