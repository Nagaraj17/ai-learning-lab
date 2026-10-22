"""Head/context/temperature diagnostics for the final-update task-piece model.

The protocol is saved before interventions. These validation diagnostics do not
retrain a model or assign semantic names to heads.
"""
import json
import hashlib
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from assignment_data import create_dataset
from assignment_evaluate import counts_for_note
from data import batch_items
from evaluate import aggregate
from experiments import diagnostic_subset
from generate import generate, inspect_prediction
from task_runtime import load_task_checkpoint
from train import atomic_json


@torch.no_grad()
def evaluate_notes(model, tokenizer, records, gates=None, mode="greedy", temperature=0.7):
    outputs, counts = [], []
    for index, record in enumerate(records):
        result = generate(model, tokenizer, record["source"], head_gates=gates, mode=mode, temperature=temperature, seed=42 + index)
        measured = counts_for_note(record, result["text"])
        counts.append(measured)
        outputs.append({"id": record["id"], "source": record["source"], "expected": record["target"], "generated": result["text"], "termination": result["termination"], "protocol_valid": result["protocol_valid"], "counts": measured})
    metrics = aggregate(counts)
    metrics.update(valid_terminations=sum(row["protocol_valid"] for row in outputs), example_ids=[row["id"] for row in records])
    inputs, targets, valid = batch_items(records, tokenizer, model.config.context)
    logits = model(inputs, valid, head_gates=gates)
    support = int((targets != -100).sum())
    loss = F.cross_entropy(logits.reshape(-1, len(tokenizer)), targets.reshape(-1), ignore_index=-100, reduction="sum")
    return {"generated_metrics": metrics, "teacher_forced": {"loss": float(loss) / support, "scored_tokens": support}, "outputs": outputs}


@torch.no_grad()
def context_probe(model, tokenizer, source, prefix, candidates=("DrugA", "DrugB"), gates=None):
    ids = tokenizer.encode(tokenizer.prompt(source) + prefix)
    logits = model(torch.tensor([ids]), head_gates=gates)[0, -1]
    probabilities = torch.softmax(logits, dim=-1)
    return {"source": source, "conditional_prefix": prefix, "input_ids": ids, "selected_token": tokenizer.tokens[int(logits.argmax())], "candidates": {word: float(probabilities[tokenizer.token_to_id[word]]) for word in candidates}, "free_generation": generate(model, tokenizer, source, head_gates=gates)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/mitra_final")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.root)
    model, tokenizer, checkpoint = load_task_checkpoint(root / "checkpoints/transformer_2_seed42/latest.pt")
    registered = json.loads((root / "results/protocol.json").read_text(encoding="utf-8"))
    if checkpoint["step"] != registered.get("steps", registered.get("stage_updates")): raise ValueError("Wait for the registered final update")
    dataset = json.loads((root / "dataset/examples.json").read_text(encoding="utf-8"))
    records = diagnostic_subset([row for row in dataset if row["split"] == "validation"], 8)
    output = root / "results/investigations"
    output.mkdir(parents=True, exist_ok=True)
    identity = {"run_id": root.name + "/transformer_2_seed42", "checkpoint_step": checkpoint["step"], "source_hash": checkpoint["source_hash"], "dataset_hash": checkpoint["dataset_hash"], "tokenizer_hash": tokenizer.fingerprint, "example_ids": [row["id"] for row in records]}
    protocol = dict(identity, diagnostic_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), single_heads=[[layer, head] for layer in range(2) for head in range(4)], pairs=[{"layer": 0, "heads": [0, 1]}, {"layer": 1, "heads": [0, 1]}], pair_selection="fixed before these interventions, not chosen from their effects", field_probe={"field": "MRN", "values": ["991723", "991724"], "scope": "one authored source-field edit, not category-wide specialization evidence"}, temperatures=[0.1, 0.7, 1.2], decoding="greedy for interventions; sampling only for temperature trials", scope="eight validation diagnostics; small category supports; no semantic head roles")
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Existing intervention protocol differs")
    atomic_json(protocol_path, protocol)
    measurements, effects = [], []
    baseline = evaluate_notes(model, tokenizer, records)
    atomic_json(output / "baseline.json", dict(identity, **baseline))
    measurements.append(dict(experiment_id="head_baseline", **{key: baseline[key] for key in ("generated_metrics", "teacher_forced")}))
    for layer in range(2):
        for head in range(4):
            gates = [1] * 4
            gates[head] = 0
            report = evaluate_notes(model, tokenizer, records, {layer: gates})
            atomic_json(output / f"layer{layer}_head{head}.json", dict(identity, disabled={"layer": layer, "heads": [head]}, **report))
            measurements.append(dict(experiment_id=f"head_{layer}_{head}", **{key: report[key] for key in ("generated_metrics", "teacher_forced")}))
            effects.append({"layer": layer, "head": head, "baseline_recall": baseline["generated_metrics"]["recall"], "ablated_recall": report["generated_metrics"]["recall"], "recall_delta": report["generated_metrics"]["recall"] - baseline["generated_metrics"]["recall"], "exact_delta": report["generated_metrics"]["exact_match"] - baseline["generated_metrics"]["exact_match"], "category_metrics": report["generated_metrics"]["categories"]})
    for pair in protocol["pairs"]:
        gates = [0 if head in pair["heads"] else 1 for head in range(4)]
        report = evaluate_notes(model, tokenizer, records, {pair["layer"]: gates})
        atomic_json(output / f"pair_layer{pair['layer']}.json", dict(identity, disabled=pair, **report))
        measurements.append(dict(experiment_id=f"pair_layer{pair['layer']}", **{key: report[key] for key in ("generated_metrics", "teacher_forced")}))
    atomic_json(output / "single_head_summary.json", dict(identity, effects=effects, pair_selection=protocol["pair_selection"]))
    atomic_json(output / "enriched_head_measurements.json", dict(identity, measurements=measurements))
    indexed = {row["experiment_id"]: row for row in measurements}
    interactions = []
    for pair in protocol["pairs"]:
        layer = pair["layer"]
        for metric in ("recall", "conditional_loss"):
            def metric_value(entry):
                return entry["teacher_forced"]["loss"] if metric == "conditional_loss" else entry["generated_metrics"]["recall"]
            base = metric_value(indexed["head_baseline"])
            singles = [metric_value(indexed[f"head_{layer}_{head}"]) - base for head in pair["heads"]]
            paired = metric_value(indexed[f"pair_layer{layer}"]) - base
            interactions.append({"layer": layer, "heads": pair["heads"], "metric": metric, "pair_delta": paired, "sum_single_deltas": sum(singles), "interaction_residual": paired - sum(singles)})
    atomic_json(output / "pair_interactions.json", dict(identity, observations=interactions, limit="Descriptive non-additivity; free-decoding thresholds and nonlinear network behavior do not establish a unique semantic mechanism"))
    # Temporary gates must not survive a request or alter parameters.
    restored = evaluate_notes(model, tokenizer, records)
    assert [row["generated"] for row in restored["outputs"]] == [row["generated"] for row in baseline["outputs"]]
    for temperature in protocol["temperatures"]:
        report = evaluate_notes(model, tokenizer, records, mode="sample", temperature=temperature)
        atomic_json(output / f"temperature_{temperature}.json", dict(identity, mode="sample", temperature=temperature, **report))
    prefix = "PATIENT [NAME] RECEIVED "
    sources = [f"PATIENT John Smith RECEIVED Drug{letter}" for letter in "AB"]
    full = [context_probe(model, tokenizer, source, prefix) for source in sources]
    assert len(full[0]["input_ids"]) == len(full[1]["input_ids"])
    assert sum(left != right for left, right in zip(full[0]["input_ids"], full[1]["input_ids"])) == 1
    embedding, _, _ = load_task_checkpoint(root / "checkpoints/embedding_only_seed42/latest.pt")
    null = [context_probe(embedding, tokenizer, source, prefix) for source in sources]
    assert null[0]["candidates"] == null[1]["candidates"]
    attention = inspect_prediction(model, tokenizer, sources[0], prefix, 0, 0)
    atomic_json(output / "context.json", dict(identity, full=full, embedding_only=null, attention=attention, note="Candidate comparison uses one changed source token and the same supplied prefix/query; free outputs are recorded separately"))
    field_probes = []
    for disabled in [None] + protocol["single_heads"]:
        gates = None
        if disabled is not None:
            layer, head = disabled
            gates = {layer: [0 if index == head else 1 for index in range(4)]}
        probes = [context_probe(model, tokenizer, f"PATIENT Olivia Martinez MRN {value} DIAGNOSIS NSCLC", "PATIENT [NAME] MRN ", ("[MRN]",), gates) for value in protocol["field_probe"]["values"]]
        assert sum(left != right for left, right in zip(probes[0]["input_ids"], probes[1]["input_ids"])) == 1
        field_probes.append({"disabled": disabled, "probes": probes})
    atomic_json(output / "field_probe.json", dict(identity, scope=protocol["field_probe"]["scope"], expected_free_text="PATIENT [NAME] MRN [MRN] DIAGNOSIS NSCLC", measurements=field_probes))
    ood = []
    for source in ("Patient Kai received DrugA yesterday.", "PATIENT Kai DIAGNOSIS UNKNOWN", "PATIENT Kai TREATMENT DrugA TREATMENT DrugB", "PATIENT Zoe DIAGNOSIS NSCLC"):
        ood.append({"source": source, "result": generate(model, tokenizer, source)})
    atomic_json(output / "ood.json", dict(identity, scope="authored qualitative probes; no invented gold accuracy", outputs=ood))
    print("Saved task-piece baseline, eight single heads, two fixed pairs, three temperatures, controlled context and qualitative OOD; restoration passed")


if __name__ == "__main__":
    main()
