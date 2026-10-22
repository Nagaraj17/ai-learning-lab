"""Version-2 sentence-anchor metrics; generation remains the shared model loop."""
import re
import time
import torch
from assignment_data import slot_values
from evaluate import aggregate
from generate import generate
from tokenizer import PLACEHOLDERS


def counts_for_note(record, prediction):
    slots = {slot["field"]: slot for slot in record["slots"]}
    tp, leaked, categories = 0, 0, {}
    for item in record["identifiers"]:
        correct = int(item["placeholder"] in slot_values(prediction, slots[item["field"]]))
        leak = bool(re.search(r"(?<![A-Za-z0-9])" + re.escape(item["value"]) + r"(?![A-Za-z0-9])", prediction, flags=re.IGNORECASE))
        tp += correct
        leaked += int(leak)
        categories[item["field"]] = {"tp": correct, "support": 1, "leaked": int(leak)}
    proposed = sum(prediction.count(marker) for marker in PLACEHOLDERS)
    preserved = sum(slot_values(prediction, slots[item["field"]]) == [item["value"]] for item in record["preserved"])
    return {"tp": tp, "fp": proposed - tp, "fn": len(record["identifiers"]) - tp, "identifiers": len(record["identifiers"]), "leaked_identifiers": leaked, "leaked_note": int(leaked > 0), "clinical_correct": preserved, "clinical_support": len(record["preserved"]), "exact": int(prediction == record["target"]), "categories": categories, "no_identifier_control": record["no_identifiers"]}


def generated_metrics_v2(model, records, tokenizer):
    outputs, counts = [], []
    if records: generate(model, tokenizer, records[0]["source"], max_new_tokens=1)
    for row in records:
        started = time.perf_counter()
        result = generate(model, tokenizer, row["source"])
        measured = counts_for_note(row, result["text"])
        counts.append(measured)
        outputs.append({"id": row["id"], "split": row["split"], "source": row["source"], "expected": row["target"], "generated": result["text"], "termination": result["termination"], "protocol_valid": result["protocol_valid"], "seconds": time.perf_counter() - started, "tokens": len(result["tokens"]), "counts": measured})
    metrics = aggregate(counts)
    metrics.update(example_ids=[row["id"] for row in records], valid_terminations=sum(row["protocol_valid"] for row in outputs), evaluation="sentence-anchor version 2; exact-surface leakage; complete clinical associations", device=str(next(model.parameters()).device))
    return {"metrics": metrics, "outputs": outputs}
