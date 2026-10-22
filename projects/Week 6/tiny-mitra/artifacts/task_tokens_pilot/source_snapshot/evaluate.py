"""Teacher-forced token metrics and separately measured generated-note metrics.

The synthetic field parser is intentionally task-specific. Exact-surface leakage
checks cannot establish clinical de-identification safety outside this grammar.
"""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
import torch
import torch.nn.functional as F
from data import batch_items, load_dataset, parse_fields
from generate import generate
from tokenizer import PLACEHOLDERS


@torch.no_grad()
def token_metrics(model, records, tokenizer, batch_size=16):
    model.eval()
    total_loss, correct, support = 0.0, 0, 0
    for start in range(0, len(records), batch_size):
        x, y, valid = batch_items(records[start:start + batch_size], tokenizer, model.config.context, next(model.parameters()).device)
        logits = model(x, valid)
        scored = y != -100
        total_loss += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100, reduction="sum"))
        correct += int(((logits.argmax(-1) == y) & scored).sum())
        support += int(scored.sum())
    return {"loss": total_loss / support if support else None, "next_token_accuracy": correct / support if support else None, "scored_tokens": support, "examples": len(records)}


def note_counts(record, prediction):
    fields = parse_fields(prediction)
    expected = Counter((item["field"], item["placeholder"]) for item in record["identifiers"])
    actual = Counter((field, value) for field, value in fields if value in PLACEHOLDERS)
    tp = sum((expected & actual).values())
    # Every printed placeholder is a proposed replacement, including malformed
    # or unassociated ones. Correct matches require the right field and type.
    proposed = sum(prediction.count(marker) for marker in PLACEHOLDERS)
    leaked, categories = 0, {}
    for item in record["identifiers"]:
        surface = item["value"]
        found = bool(re.search(r"(?<![A-Za-z0-9])" + re.escape(surface) + r"(?![A-Za-z0-9])", prediction, flags=re.IGNORECASE))
        leaked += int(found)
        category = categories.setdefault(item["field"], {"tp": 0, "support": 0, "leaked": 0})
        category["support"] += 1
        category["leaked"] += int(found)
    # Counter intersection prevents duplicate gold/printed fields inflating TP.
    for (field, _), count in (expected & actual).items():
        categories[field]["tp"] += count
    clinical = sum(fields.count((item["field"], item["value"])) == 1 for item in record["preserved"])
    return {"tp": tp, "fp": proposed - tp, "fn": len(record["identifiers"]) - tp, "identifiers": len(record["identifiers"]), "leaked_identifiers": leaked, "leaked_note": int(leaked > 0), "clinical_correct": clinical, "clinical_support": len(record["preserved"]), "exact": int(prediction == record["target"]), "categories": categories, "no_identifier_control": record["no_identifiers"]}


def aggregate(counts):
    summed = {key: sum(item[key] for item in counts) for key in ("tp", "fp", "fn", "identifiers", "leaked_identifiers", "leaked_note", "clinical_correct", "clinical_support", "exact")}
    precision = summed["tp"] / (summed["tp"] + summed["fp"]) if summed["tp"] + summed["fp"] else None
    recall = summed["tp"] / summed["identifiers"] if summed["identifiers"] else None
    f1 = 2 * summed["tp"] / (2 * summed["tp"] + summed["fp"] + summed["fn"]) if 2 * summed["tp"] + summed["fp"] + summed["fn"] else None
    categories = {}
    for item in counts:
        for field, measures in item["categories"].items():
            total = categories.setdefault(field, {"tp": 0, "support": 0, "leaked": 0})
            for key, value in measures.items():
                total[key] += value
    for measures in categories.values():
        measures["recall"] = measures["tp"] / measures["support"]
    controls = [item for item in counts if item["no_identifier_control"]]
    return dict(summed, examples=len(counts), precision=precision, recall=recall, f1=f1, exact_match=summed["exact"] / len(counts) if counts else None, identifier_leakage=summed["leaked_identifiers"] / summed["identifiers"] if summed["identifiers"] else None, note_leakage=summed["leaked_note"] / len(counts) if counts else None, clinical_preservation=summed["clinical_correct"] / summed["clinical_support"] if summed["clinical_support"] else None, categories=categories, controls=len(controls), control_over_redactions=sum(item["fp"] for item in controls))


def generated_metrics(model, records, tokenizer, head_gates=None, mode="greedy", temperature=0.7, seed=42):
    rows, counts, seconds, token_count = [], [], 0.0, 0
    device = next(model.parameters()).device
    # Warm-up excludes model loading and makes measured generation comparable.
    if records:
        generate(model, tokenizer, records[0]["source"], max_new_tokens=1, head_gates=head_gates)
    for index, record in enumerate(records):
        if device.type == "cuda": torch.cuda.synchronize()
        started = time.perf_counter()
        result = generate(model, tokenizer, record["source"], head_gates=head_gates, mode=mode, temperature=temperature, seed=seed + index)
        if device.type == "cuda": torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        measured = note_counts(record, result["text"])
        counts.append(measured)
        seconds += elapsed
        token_count += len(result["tokens"])
        rows.append({"id": record["id"], "split": record["split"], "source": record["source"], "expected": record["target"], "generated": result["text"], "termination": result["termination"], "protocol_valid": result["protocol_valid"], "seconds": elapsed, "tokens": len(result["tokens"]), "counts": measured})
    summary = aggregate(counts)
    summary.update(seconds=seconds, generated_tokens=token_count, seconds_per_note=seconds / len(rows) if rows else None, valid_terminations=sum(row["protocol_valid"] for row in rows), mode=mode, temperature=temperature if mode == "sample" else None, sampling_seed=seed, device=str(device), coverage="declared subset; see IDs", example_ids=[row["id"] for row in rows])
    return {"metrics": summary, "outputs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--output", default="artifacts/results/evaluation.json")
    args = parser.parse_args()
    from train import load_checkpoint
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint)
    records = [record for record in load_dataset() if record["split"] == args.split]
    selected = records[:args.limit]
    result = generated_metrics(model, selected, tokenizer)
    result.update(teacher_forced=token_metrics(model, records, tokenizer), run_id=checkpoint["run_id"], dataset_hash=checkpoint["dataset_hash"], tokenizer_hash=tokenizer.fingerprint, selected_step=checkpoint["step"], source_hash=checkpoint["source_hash"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
