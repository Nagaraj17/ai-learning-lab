"""Additional measurements of existing checkpoints; no model implementation here.

Keep token-conditional probes, teacher-forced loss and free generation separate.
These functions enrich saved interventions without changing trained weights.
"""
from collections import Counter
import torch
import torch.nn.functional as F
from data import batch_items, parse_fields
from tokenizer import PLACEHOLDERS


@torch.no_grad()
def masked_token_metrics(model, records, tokenizer, head_gates=None, batch_size=8):
    """Measure output-only token loss with an optional context-head intervention."""
    model.eval()
    loss_sum, correct, support = 0.0, 0, 0
    for start in range(0, len(records), batch_size):
        x, y, valid = batch_items(records[start:start + batch_size], tokenizer, model.config.context, next(model.parameters()).device)
        logits = model(x, valid, head_gates=head_gates)
        scored = y != -100
        loss_sum += float(F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum"))
        correct += int(((logits.argmax(-1) == y) & scored).sum())
        support += int(scored.sum())
    return {"loss": loss_sum / support if support else None, "next_token_accuracy": correct / support if support else None, "scored_tokens": support, "examples": len(records)}


@torch.no_grad()
def conditional_probe(model, tokenizer, source, prefix, expected):
    """Record probabilities/logits at the exact query predicting after prefix."""
    model.eval()
    text = tokenizer.prompt(source) + prefix
    ids, offsets = tokenizer.encode_with_offsets(text)
    logits = model(torch.tensor([ids], device=next(model.parameters()).device), capture=True)[0, -1]
    probabilities = torch.softmax(logits, dim=-1)
    expected_ids = tokenizer.encode(expected)
    if len(expected_ids) != 1:
        raise ValueError("Probe expects one actual next token")
    candidates = {token: {"logit": float(logits[tokenizer.token_to_id[token]]), "probability": float(probabilities[tokenizer.token_to_id[token]])} for token in ("A", "B", "[NAME]", "[MRN]")}
    attention = [matrix[0, :, -1].tolist() for matrix in model.attention_matrices()]
    return {"source": source, "conditional_prefix": prefix, "query_index": len(ids) - 1, "input_ids": ids, "input_tokens": [tokenizer.tokens[index] for index in ids], "offsets": offsets, "expected": expected, "chosen": tokenizer.tokens[int(logits.argmax())], "expected_probability": float(probabilities[expected_ids[0]]), "expected_logit": float(logits[expected_ids[0]]), "candidates": candidates, "attention_rows": attention, "scope": "supplied prefix; not a freely generated prefix"}


def category_precision(records, outputs):
    """Count proposed placeholders by their parsed field, including wrong types.

    This preserves the original global TP/FP/FN definition. Unsupported categories
    report recall=None. Unassociated placeholders are separate false positives.
    """
    by_id = {record["id"]: record for record in records}
    totals = {}
    for row in outputs:
        record = by_id[row["id"]]
        expected = Counter((item["field"], item["placeholder"]) for item in record["identifiers"])
        actual = Counter((field, value) for field, value in parse_fields(row["generated"]) if value in PLACEHOLDERS)
        correct = expected & actual
        proposed, support, matched = Counter(), Counter(), Counter()
        for field, value in parse_fields(row["generated"]):
            proposed[field] += sum(value.count(marker) for marker in PLACEHOLDERS)
        all_proposed = sum(row["generated"].count(marker) for marker in PLACEHOLDERS)
        proposed["UNASSOCIATED"] += all_proposed - sum(proposed.values())
        for (field, _), count in expected.items(): support[field] += count
        for (field, _), count in correct.items(): matched[field] += count
        for field in set(proposed) | set(support):
            total = totals.setdefault(field, {"tp": 0, "fp": 0, "fn": 0, "support": 0, "proposed": 0})
            total["tp"] += matched[field]
            total["fp"] += proposed[field] - matched[field]
            total["fn"] += support[field] - matched[field]
            total["support"] += support[field]
            total["proposed"] += proposed[field]
    for total in totals.values():
        total["precision"] = total["tp"] / total["proposed"] if total["proposed"] else None
        total["recall"] = total["tp"] / total["support"] if total["support"] else None
    return totals
