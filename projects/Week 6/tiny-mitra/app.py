"""Local Gradio UI: real checkpoints and saved evidence, no training on clicks."""
import argparse
import json
import threading
from functools import lru_cache
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import gradio as gr
from config import ARCHITECTURES
from generate import generate, inspect_prediction
from train import load_checkpoint

LOCK = threading.Lock()
ROOT = Path(__file__).parent
CHECKPOINTS = ROOT / "artifacts/checkpoints"
RESULTS = ROOT / "artifacts/results"
ASSIGNMENT_ROOT = ROOT / "artifacts/assignment_v2"
TASK_ROOT = ROOT / "artifacts/mitra_final"
REPEAT_MODELS = {f"final_seed{seed}/transformer_2": seed for seed in (7, 123)}
MODEL_OPTIONS = tuple("task_tokens/" + name for name in ARCHITECTURES) + tuple(REPEAT_MODELS) + tuple("assignment_v2/" + name for name in ARCHITECTURES) + ARCHITECTURES + ("transformer_2_latest", "learning_pilot")


def display_number(value):
    """Readable UI decimals; full-precision measurements remain in saved JSON."""
    return round(value, 3) if isinstance(value, float) else value


@lru_cache(maxsize=12)
def cached_model(path_string, modified_ns, size, task_tokens=False):
    # Cache identity changes when a checkpoint is replaced during training.
    if task_tokens:
        from task_runtime import load_task_checkpoint
        return load_task_checkpoint(path_string)
    return load_checkpoint(path_string)


def selected_model(architecture):
    if architecture not in MODEL_OPTIONS:
        raise ValueError("Unknown architecture")
    assignment_series = architecture.startswith("assignment_v2/")
    task_series = architecture.startswith("task_tokens/") or architecture in REPEAT_MODELS
    seed = REPEAT_MODELS.get(architecture, 42)
    expected_architecture = architecture.split("/", 1)[1] if assignment_series or task_series else "transformer_2" if architecture in ("transformer_2_latest", "learning_pilot") else architecture
    filename = "latest.pt" if task_series or architecture == "transformer_2_latest" else "best.pt"
    directory = "pilot_overfit" if architecture == "learning_pilot" else f"{expected_architecture}_seed{seed}"
    root = TASK_ROOT / "checkpoints" if task_series else ASSIGNMENT_ROOT / "checkpoints" if assignment_series else CHECKPOINTS
    path = root / directory / filename
    if not path.exists():
        raise ValueError(f"No trained checkpoint for {architecture}. Train/download it first.")
    stat = path.stat()
    model, tokenizer, checkpoint = cached_model(str(path), stat.st_mtime_ns, stat.st_size, task_series)
    if model.config.architecture != expected_architecture:
        raise ValueError("Checkpoint architecture does not match selection")
    if architecture in REPEAT_MODELS and (checkpoint["step"] != 4000 or checkpoint["protocol"]["seed"] != seed):
        raise ValueError("This registered repeat has not completed its 4,000 updates yet")
    if architecture != "learning_pilot":
        manifest_path = TASK_ROOT / "dataset/manifest.json" if task_series else ASSIGNMENT_ROOT / "dataset/manifest.json" if assignment_series else ROOT / "artifacts/dataset/manifest.json"
        if not manifest_path.exists() or checkpoint["dataset_hash"] != json.loads(manifest_path.read_text(encoding="utf-8"))["dataset_hash"]:
            raise ValueError("Checkpoint dataset does not match the selected study")
    lineage_path = path.parent / "initialization.json"
    if task_series and lineage_path.exists():
        checkpoint = dict(checkpoint, lineage=json.loads(lineage_path.read_text(encoding="utf-8")))
    return model, tokenizer, checkpoint


def compare(source, architectures, mode="greedy", temperature=0.7):
    if not source.strip() or not architectures:
        raise gr.Error("Enter a synthetic note and choose at least one architecture")
    rendered, traces = [], {}
    with LOCK:
        for architecture in architectures:
            try:
                model, tokenizer, checkpoint = selected_model(architecture)
                result = generate(model, tokenizer, source, mode=mode, temperature=float(temperature))
                selection = "separately labelled final-corpus development repeat; fixed 4,000th update; primary seed42 is unchanged" if architecture in REPEAT_MODELS else "fixed task pieces with character fallback; latest saved update; read the registered study budget" if architecture.startswith("task_tokens/") else "assignment-format version 2; minimum validation loss; read its separate results" if architecture.startswith("assignment_v2/") else "latest update; exploratory, not the registered benchmark selection" if architecture == "transformer_2_latest" else "separate four-note training pilot, not the 1,000-example benchmark" if architecture == "learning_pilot" else "version-1 flat-field study; minimum validation loss"
                if "lineage" in checkpoint:
                    selection += f". Continued from our own {checkpoint['lineage']['parent_updates']}-update model; this stage has {checkpoint['step']} updates, {checkpoint['lineage']['parent_updates'] + checkpoint['step']} cumulative"
                rendered.append(f"### {architecture}\n\n```text\n{result['text']}\n```\nStop: {result['termination']}; checkpoint update: {checkpoint['step']}; training seed: {checkpoint['protocol']['seed']}. Selection: {selection}.")
                traces[architecture] = result
            except ValueError as error:
                rendered.append(f"### {architecture}\n\n{error}")
    return "\n\n".join(rendered), traces


def inspect(architecture, source, output_prefix, layer, head):
    try:
        with LOCK:
            model, tokenizer, _ = selected_model(architecture)
            result = inspect_prediction(model, tokenizer, source, output_prefix, int(layer), int(head))
        figure, axis = plt.subplots(figsize=(12, 2.7))
        axis.imshow([result["attention"]], aspect="auto", cmap="Blues", vmin=0)
        stride = max(1, len(result["input_tokens"]) // 24)
        positions = list(range(0, len(result["input_tokens"]), stride))
        axis.set_xticks(positions, [repr(result["input_tokens"][index]) for index in positions], rotation=60, ha="right")
        axis.set_yticks([0], ["final query"])
        axis.set_title(f"Layer {layer}, head {head}: next token {result['next_token']!r}; conditional prefix")
        figure.tight_layout()
        return figure, result
    except ValueError as error:
        raise gr.Error(str(error)) from error


def live_tokens(source, architecture, mode="greedy", temperature=0.7):
    """Send each prediction to Gradio as soon as its forward pass completes."""
    from live_generation import stream_generate
    try:
        with LOCK:
            model, tokenizer, checkpoint = selected_model(architecture)
            for frame in stream_generate(model, tokenizer, source, mode=mode, temperature=float(temperature)):
                caption = f"### {architecture}\nUpdate {checkpoint['step']}; training seed {checkpoint['protocol']['seed']}. {frame['termination']}\n\n```text\n{frame['text']}\n```"
                yield caption, frame
    except ValueError as error:
        raise gr.Error(str(error)) from error


def task_token_metrics():
    path = TASK_ROOT / "results/final_test/token_metrics_audit.json"
    if not path.exists():
        return [], "Recomputed final token metrics are pending."
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise gr.Error("Final token metrics did not pass review")
    rows = [[r["architecture"], r["split"], r["examples"], r["scored_tokens"], round(r["loss"], 4), round(r["next_token_accuracy"], 4)] for r in report["rows"]]
    return rows, "Primary seed42, fixed update4000. Teacher forcing supplies the correct output prefix. Loss and accuracy score output/EOS tokens; these differ from free-generation exact match. All32 saved rows were recomputed from the matching weights."


def intervene(architecture, source, layer, head):
    try:
        with LOCK:
            model, tokenizer, _ = selected_model(architecture)
            if not model.blocks:
                raise ValueError("Choose a Transformer architecture for block/head intervention")
            layer, head = int(layer), int(head)
            if not 0 <= layer < len(model.blocks) or not 0 <= head < model.config.n_heads:
                raise ValueError("Layer/head outside model")
            gates = [1] * model.config.n_heads
            gates[head] = 0
            baseline = generate(model, tokenizer, source)
            ablated = generate(model, tokenizer, source, head_gates={layer: gates})
        return {"baseline": baseline, "ablated": ablated, "disabled": {"layer": layer, "head": head}, "note": "Live input has no gold label; these outputs alone provide no precision/recall score. Gates are local to this request."}
    except ValueError as error:
        raise gr.Error(str(error)) from error


def intervention_view(architecture, source, layer, head):
    result = intervene(architecture, source, layer, head)
    text = "\n\n".join(f"### {name.capitalize()}\n\n```text\n{result[name]['text']}\n```\nStop: {result[name]['termination']}" for name in ("baseline", "ablated"))
    return text + "\n\n" + result["note"], result


def saved_evidence():
    path = RESULTS / "benchmark_summary.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    repeated = RESULTS / "repeats_summary.json"
    if repeated.exists():
        rows += json.loads(repeated.read_text(encoding="utf-8"))
    columns = ["architecture", "seed", "validation_loss", "validation_token_accuracy", "recall", "precision", "f1", "exact_match", "note_leakage", "clinical_preservation", "parameters", "examples"]
    table = [[display_number(row.get(column)) for column in columns] for row in rows]
    figure, axis = plt.subplots(figsize=(10, 4))
    for directory in CHECKPOINTS.glob("*_seed42"):
        if directory.name.removesuffix("_seed42") not in ARCHITECTURES: continue
        history = directory / "history.json"
        if history.exists():
            data = json.loads(history.read_text(encoding="utf-8"))
            axis.plot([row["step"] for row in data], [row["validation"]["loss"] for row in data], label=directory.name.replace("_seed42", ""))
    axis.set(xlabel="Optimizer updates", ylabel="Token-weighted validation loss")
    axis.set_title("Primary comparison: initialization seed 42")
    if axis.lines: axis.legend(fontsize=7)
    else: axis.text(0.5, 0.5, "No saved training curves yet", ha="center")
    figure.tight_layout()
    path = RESULTS / "investigations/single_head_summary.json"
    heads = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "Head experiments not run yet"}
    return table, figure, heads


def final_partition_evidence():
    path = RESULTS / "final_test/summary.json"
    if not path.exists():
        return [], "Final-partition evaluation is pending."
    rows = json.loads(path.read_text(encoding="utf-8"))
    columns = ("architecture", "split", "examples", "recall", "precision", "exact_match", "clinical_preservation", "note_leakage")
    return [[display_number(row.get(key)) for key in columns] for row in rows], f"{len(rows)} of 32 primary architecture/partition rows saved. Each completed partition contains 50 notes. Controls share a layout; unseen names/templates refer to the declared non-control conditions."


def checkpoint_selection_evidence():
    paths = [RESULTS / "transformer_2_seed42_validation.json", RESULTS / "latest_checkpoint_diagnostic.json"]
    if not all(path.exists() for path in paths):
        return "Checkpoint-selection diagnostics are pending."
    selected, latest = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if selected["metrics"]["example_ids"] != latest["metrics"]["example_ids"]:
        raise gr.Error("Checkpoint diagnostic IDs differ; comparison requires matched examples")
    rows = ["| Checkpoint | Exact notes | Correct identifier replacements | Preserved clinical fields |", "| --- | ---: | ---: | ---: |"]
    for name, report in (("Best validation loss", selected), ("Latest training update (exploratory)", latest)):
        metrics = report["metrics"]
        rows.append(f"| {name} | {metrics['exact']}/{metrics['examples']} | {metrics['tp']}/{metrics['identifiers']} | {metrics['clinical_correct']}/{metrics['clinical_support']} |")
    return "\n".join(rows) + "\n\nSame 12 validation diagnostics. This exploratory selection comparison does not replace the registered architecture benchmark or establish final-test quality."


def saved_head_evidence(metric="recall", study="original_characters"):
    """Chart measured recall changes; absence of a report is not zero effect."""
    if study not in ("task_tokens", "original_characters"): raise gr.Error("Unknown saved head study")
    report_root = TASK_ROOT / "results" if study == "task_tokens" else RESULTS
    path = report_root / "investigations/single_head_summary.json"
    if not path.exists():
        raise gr.Error("Head experiments are not complete yet")
    report = json.loads(path.read_text(encoding="utf-8"))
    effects = report["effects"]
    figure, axis = plt.subplots(figsize=(9, 4))
    labels = [f"L{row['layer']} H{row['head']}" for row in effects]
    deltas = [row["recall_delta"] for row in effects]
    if metric not in ("recall", "conditional_loss", "clinical_preservation"):
        raise gr.Error("Unknown head metric")
    enriched_path = report_root / "investigations/enriched_head_measurements.json"
    if metric != "recall":
        if not enriched_path.exists():
            raise gr.Error("Additional head measurements are pending")
        enriched = json.loads(enriched_path.read_text(encoding="utf-8"))
        indexed = {entry["experiment_id"]: entry for entry in enriched["measurements"]}
        def value(entry):
            return entry["teacher_forced"]["loss"] if metric == "conditional_loss" else entry["generated_metrics"][metric]
        baseline = value(indexed["head_baseline"])
        deltas = [value(indexed[f"head_{row['layer']}_{row['head']}"]) - baseline for row in effects]
        report["additional_measurements"] = enriched
    axis.bar(labels, deltas)
    axis.axhline(0, color="black", linewidth=0.7)
    direction = "negative is improvement" if metric == "conditional_loss" else "positive is improvement"
    axis.set(ylabel=f"{metric}: ablated minus baseline; {direction}", title=f"Saved {report['run_id']}, update {report['checkpoint_step']}: {len(report['example_ids'])} diagnostic notes")
    figure.tight_layout()
    return figure, report


def reviewed_findings():
    path = ROOT / "reports/hypothesis_reviews.json"
    if not path.exists():
        return "Experiment interpretations are awaiting evidence review. No hypothesis is marked supported merely because the code exists."
    reports = json.loads(path.read_text(encoding="utf-8"))
    lines = []
    for report in reports:
        lines.append(f"### {report['hypothesis_id']}: {report['question']}")
        for field in ("change", "observation", "verdict", "interpretation", "limitations"):
            lines.append(f"**{field.capitalize()}:** {report[field]}")
    return "\n\n".join(lines)


def assignment_evidence(root=None, label="Instructor-format character study"):
    """Keep the corrected grammar's measurements apart from the first study."""
    root = ASSIGNMENT_ROOT if root is None else root
    path = root / "results/summary.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    columns = ("architecture", "selected_step", "updates", "parameters", "validation_loss", "validation_token_accuracy", "diagnostic_exact", "diagnostic_recall", "diagnostic_clinical_preservation", "acceptance_correct_notes", "acceptance_notes")
    figure, axis = plt.subplots(figsize=(10, 4))
    for architecture in ARCHITECTURES:
        history = root / f"checkpoints/{architecture}_seed42/history.json"
        if history.exists():
            values = json.loads(history.read_text(encoding="utf-8"))
            axis.plot([row["step"] for row in values], [row["validation"]["loss"] for row in values], label=architecture)
    protocol_path = root / "results/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else {}
    budget = protocol.get("steps", protocol.get("stage_updates", 2000))
    axis.set(xlabel="Optimizer updates in this stage", ylabel="Token-weighted validation loss", title=label + f": {budget:,}-update stage")
    if axis.lines: axis.legend(fontsize=7)
    else: axis.text(0.5, 0.5, "No saved validation curves yet", ha="center")
    figure.tight_layout()
    status = f"{len(rows)} of 8 runs have saved development-generation results. Curves may include runs still training. Twenty-four validation diagnostics and eleven authored instructor/clinical prompts are development evidence, not final-test accuracy. Compare architectures within this series. Token loss has different units under different tokenizers and must not be used as a cross-tokenizer quality ranking."
    protocol_path = root / "results/protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if "parent_updates" in protocol:
            status += f" This continued stage uses {protocol['parent_updates']} parent updates plus {protocol['stage_updates']} broader-input updates ({protocol['cumulative_updates']} cumulative); optimizer and sampler were reset for the new stage."
    return [[display_number(row.get(column)) for column in columns] for row in rows], figure, status


def assignment_acceptance(architecture, root=None, label="Instructor-format character study"):
    root = ASSIGNMENT_ROOT if root is None else root
    if architecture not in ARCHITECTURES: raise gr.Error("Choose a registered architecture")
    path = root / f"results/{architecture}_acceptance.json"
    if not path.exists(): return "This run has not saved its authored-prompt generations yet."
    report = json.loads(path.read_text(encoding="utf-8"))
    lineage_note = f" Parent updates: {report['parent_updates']}; cumulative: {report['cumulative_updates']}." if "parent_updates" in report else ""
    lines = [f"### {label}: {architecture}, selected update {report['selected_step']}" + lineage_note, "Authored development prompts. Expected labels and actual free generations are shown separately; no answer is substituted into the model output."]
    for row in report["outputs"]:
        lines.append(f"**{row['id']}** — stop: {row['termination']}\n\n```text\nSource:    {row['source']}\nExpected:  {row['expected']}\nGenerated: {row['generated']}\n```")
    return "\n\n".join(lines)


def task_evidence():
    table, figure, status = assignment_evidence(TASK_ROOT, "Fixed task pieces with character fallback")
    path = TASK_ROOT / "results/protocol.json"
    parent_updates = json.loads(path.read_text(encoding="utf-8")).get("parent_updates", 0) if path.exists() else 0
    for row in table: row.insert(3, row[2] + parent_updates)
    return table, figure, status


def task_acceptance(architecture):
    return assignment_acceptance(architecture, TASK_ROOT, "Fixed task pieces with character fallback")


def task_repeat_evidence():
    path = TASK_ROOT / "results/repeats_summary.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    columns = ("architecture", "seed", "updates", "diagnostic_exact", "diagnostic_recall", "diagnostic_clinical_preservation", "acceptance_correct_notes", "acceptance_notes")
    return [[display_number(row.get(key)) for key in columns] for row in rows], f"{len(rows)}/8 repeat runs complete. Seeds7 and123 vary initialization and minibatch order on the same frozen data. Primary seed42 stays unchanged. Shared development notes are not independent datasets. Live full-model repeats use explicitly named final_seed7/ and final_seed123/ choices. No final-test outcomes choose a demo seed."


def task_study_description():
    path = TASK_ROOT / "results/protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if "parent_updates" in protocol:
        return f"## Fixed task pieces — broader-input continued stage\nEach architecture starts from its own {protocol['parent_updates']:,}-update model and trains another {protocol['stage_updates']:,} updates with broader name lengths and identifier digits. The new stage resets optimizer and sampler; lineage is saved. Added training and input coverage both change. Character studies below preserve their original settings and selection rules."
    budget = protocol.get("steps", "pending")
    return f"## Main Mitra — one corpus, random initial weights\nThe final comparison uses one saved corpus of 1,000 synthetic examples and {budget} updates per architecture. It starts from random weights, with no weights from earlier experiments. Fixed task pieces retain character fallback. Vocabulary/sequence lengths differ from character studies; cross-tokenizer loss is not a quality ranking."


def task_final_partitions():
    path = TASK_ROOT / "results/final_test/summary.json"
    if not path.exists(): return [], "Final single-corpus held-out evaluation is pending its saved protocol review/release."
    rows = json.loads(path.read_text(encoding="utf-8"))
    columns = ("architecture", "split", "examples", "recall", "precision", "exact_match", "clinical_preservation", "note_leakage")
    return [[display_number(row.get(key)) for key in columns] for row in rows], f"{len(rows)} of 32 final model/partition rows saved, each using all 50 notes. The 200 distinct test notes are reused across eight models; they are not 1,600 independent cases. Read the declared entity/layout pools and shared-control limitations."


def demo_final_partitions():
    path = TASK_ROOT / "results/demo_final_test/summary.json"
    if not path.exists(): return [], "The separately selected demo's held-out measurement is pending. Its seed is selected from development evidence only."
    report = json.loads(path.read_text(encoding="utf-8"))
    columns = ("split", "examples", "recall", "precision", "exact_match", "clinical_preservation", "note_leakage")
    return [[display_number(row.get(key)) for key in columns] for row in report["rows"]], f"{report.get('model_id', 'none')}: {len(report['rows'])}/4 complete partitions. Selection was fixed before these outputs. These are the same 200 test notes used by the primary comparison, not additional independent cases. Status: {report['status']}."


def context_evidence():
    path = TASK_ROOT / "results/investigations/architecture_context.json"
    if not path.exists(): return [], {}, "The matched source-pair experiment has not run for this study."
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = [[row["architecture"], *[item["selected_token"] for item in row["observations"]], row["both_conditional_and_free_correct"], display_number(row["maximum_probability_change"])] for row in report["models"]]
    return rows, report, "Same-length sources differ only in DrugA/DrugB. The supplied output prefix ends `PATIENT [NAME] RECEIVED `, including its final space. Conditional choices and free generations are separate checks. Read the raw report for probabilities and actual free outputs; this one development pair does not establish broad generalization."


def final_reviewed_findings():
    path = ROOT / "reports/final_hypothesis_reviews.json"
    if not path.exists(): return "Final-corpus interpretations are awaiting author review. Measured outputs remain available; earlier-study findings are below in their own section."
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "passed": return "Final interpretations are still under review."
    lines = ["Final corpus, primary seed42. Repeats and the separately selected demo retain their declared scope."]
    for row in report["hypotheses"]:
        lines.append(f"### {row['hypothesis_id']}: {row['question']}")
        for field in ("change", "observation", "verdict", "interpretation", "limitations"):
            lines.append(f"**{field.capitalize()}:** {row[field]}")
    return "\n\n".join(lines)


def build_app():
    columns = ["architecture", "seed", "val loss", "val token accuracy", "recall", "precision", "F1", "exact match", "note leakage", "clinical preservation", "parameters", "examples"]
    with gr.Blocks(title="Tiny Mitra — learn, compare, investigate", analytics_enabled=False) as demo:
        gr.Markdown("# Tiny Mitra\nA small GPT assembled from PyTorch primitives. Synthetic educational task. Saved benchmarks and live generations are labeled separately.")
        with gr.Tab("Compare models"):
            source = gr.Textbox(label="Synthetic input note", value="PATIENT Jo TREATMENT DrugA", lines=3)
            architectures = gr.Dropdown(list(MODEL_OPTIONS), value=["learning_pilot"], multiselect=True, label="Trained models/checkpoints to compare")
            gr.Examples(examples=[["PATIENT Jo TREATMENT DrugA", ["learning_pilot"]], ["PATIENT Jo TREATMENT DrugB", ["learning_pilot"]], ["PATIENT John Smith VISITED Dr Jones ON January 12", ["task_tokens/transformer_2"]], ["PATIENT Olivia Martinez MRN 991723 DIAGNOSIS NSCLC", ["task_tokens/transformer_2"]]], inputs=[source, architectures], label="Teaching pilot and fixed-task-piece experiment")
            with gr.Accordion("Earlier character-study examples", open=False):
                gr.Examples(examples=[["PATIENT John Smith VISITED Dr Jones ON January 12", ["assignment_v2/transformer_2"]], ["PATIENT Kevin Black LIVES 412 Main Street Dallas Texas", ["assignment_v2/transformer_2"]], ["PATIENT John Lee DIAGNOSIS NSCLC BIOMARKER EGFR TREATMENT DrugA", ["transformer_2", "transformer_2_latest"]]], inputs=[source, architectures], label="Preserved character studies; their outputs may be wrong")
            mode = gr.Radio(["greedy", "sample"], value="greedy", label="Decoding method")
            temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Sampling temperature (ignored by greedy)")
            button = gr.Button("Generate and compare")
            outputs = gr.Markdown()
            with gr.Accordion("Actual generated token trace", open=False):
                trace = gr.JSON()
            button.click(compare, [source, architectures, mode, temperature], [outputs, trace], api_name="compare")
            with gr.Accordion("Watch actual next-token predictions", open=False):
                live_model = gr.Dropdown(list(MODEL_OPTIONS), value="final_seed7/transformer_2", label="One trained checkpoint")
                live_button = gr.Button("Generate tokens one at a time")
                live_text = gr.Markdown()
                live_trace = gr.JSON(label="Growing actual token trace")
                live_button.click(live_tokens, [source, live_model, mode, temperature], [live_text, live_trace], api_name="live_tokens")
            gr.Markdown("Live text has no reference answer, so no live precision/recall is invented. Missing checkpoints are shown explicitly.")
            gr.Markdown("`transformer_2_latest` uses the same architecture at its last update. It is an exploratory checkpoint comparison; saved architecture benchmarks use their best-validation checkpoints.")
            gr.Markdown("`learning_pilot` is a smaller two-block model trained on four short notes. Its successful demonstration is separate from the broader benchmark and its generalization limits.")
            gr.Markdown("`assignment_v2/...` models use the corrected instructor sentence grammar and a separate training series. Unprefixed architecture names refer to the earlier flat-field study. A checkpoint may become available before its full training run has finished; the displayed update identifies what is loaded.")
            gr.Markdown("`task_tokens/...` models use fixed schema/clinical token pieces plus character spelling for other text, with a separately registered final-update comparison. Their generations and metrics must be measured; changing the tokenizer alone does not establish success.")
            gr.Markdown("`final_seed7/transformer_2` and `final_seed123/transformer_2` are separately identified development repeats on the same final corpus. They load only after all 4,000 updates. Seed42 remains the primary architecture benchmark; a repeat is not silently substituted for it.")
        with gr.Tab("Inspect heads"):
            architecture = gr.Dropdown(list(MODEL_OPTIONS), value="task_tokens/transformer_2", label="Model/checkpoint")
            note = gr.Textbox(value="PATIENT John Smith RECEIVED DrugA", label="Synthetic source")
            prefix = gr.Textbox(value="PATIENT [NAME] RECEIVED ", label="Conditional output prefix (supplied, not freely generated)")
            gr.Examples(examples=[["task_tokens/transformer_2", "PATIENT John Smith RECEIVED DrugA", "PATIENT [NAME] RECEIVED "], ["task_tokens/transformer_2", "PATIENT John Smith RECEIVED DrugB", "PATIENT [NAME] RECEIVED "], ["learning_pilot", "PATIENT Jo TREATMENT DrugA", "PATIENT [NAME] TREATMENT Drug"]], inputs=[architecture, note, prefix], label="Conditional probes use the selected tokenizer's piece boundaries")
            layer = gr.Slider(0, 1, value=0, step=1, label="Block index (starts at 0)")
            head = gr.Slider(0, 3, value=0, step=1, label="Head index (starts at 0)")
            inspect_button = gr.Button("Inspect next prediction")
            heatmap = gr.Plot()
            with gr.Accordion("Token offsets and raw attention values", open=False):
                details = gr.JSON()
            inspect_button.click(inspect, [architecture, note, prefix, layer, head], [heatmap, details], api_name="inspect")
            ablate_button = gr.Button("Generate with this head disabled")
            intervention_text = gr.Markdown()
            with gr.Accordion("Raw intervention token traces", open=False):
                intervention = gr.JSON()
            ablate_button.click(intervention_view, [architecture, note, layer, head], [intervention_text, intervention], api_name="intervene")
            head_button = gr.Button("Load saved head-ablation comparison")
            head_metric = gr.Radio(["recall", "conditional_loss", "clinical_preservation"], value="recall", label="Saved intervention metric")
            head_study = gr.Radio(["task_tokens", "original_characters"], value="task_tokens", label="Saved head experiment study (read its checkpoint caption)")
            head_chart = gr.Plot()
            with gr.Accordion("Category metrics and exact diagnostic examples", open=False):
                head_evidence = gr.JSON()
            head_button.click(saved_head_evidence, [head_metric, head_study], [head_chart, head_evidence], api_name="head_evidence")
            gr.Markdown("Attention shows routing for this prediction. Ablation measures dependence under an intervention. Neither automatically names a head's semantic role.")
        with gr.Tab("Compare components"):
            gr.Markdown(task_study_description())
            task_table = gr.Dataframe(headers=["architecture", "selected stage update", "stage updates", "cumulative updates", "parameters", "val loss", "val token accuracy", "diagnostic exact", "diagnostic recall", "diagnostic clinical preservation", "authored exact notes", "authored notes"], interactive=False)
            task_curves = gr.Plot()
            task_status = gr.Markdown()
            task_refresh = gr.Button("Load task-piece results")
            task_refresh.click(task_evidence, [], [task_table, task_curves, task_status], api_name="task_evidence")
            with gr.Accordion("Final test loss and next-token accuracy", open=False):
                token_refresh = gr.Button("Load recomputed test token metrics")
                token_table = gr.Dataframe(headers=["architecture", "partition", "notes", "scored output/EOS tokens", "loss", "next-token accuracy"], interactive=False)
                token_status = gr.Markdown()
                token_refresh.click(task_token_metrics, [], [token_table, token_status], api_name="task_token_metrics")
            with gr.Accordion("Controlled source dependence: one head versus several", open=False):
                context_refresh = gr.Button("Load matched source-pair results")
                context_table = gr.Dataframe(headers=["architecture", "DrugA source: next token", "DrugB source: next token", "both conditional and free correct", "max probability change"], interactive=False)
                context_status = gr.Markdown()
                with gr.Accordion("Exact sources, probabilities and free generations", open=False):
                    context_raw = gr.JSON()
                context_refresh.click(context_evidence, [], [context_table, context_raw, context_status], api_name="context_evidence")
            final_findings_button = gr.Button("Read final-corpus reviewed findings")
            final_findings = gr.Markdown()
            final_findings_button.click(final_reviewed_findings, [], final_findings, api_name="final_findings")
            with gr.Accordion("Final-corpus training-seed sensitivity", open=False):
                repeat_refresh = gr.Button("Load final repeat results")
                repeat_table = gr.Dataframe(headers=["architecture", "training seed", "updates", "diagnostic exact", "diagnostic recall", "diagnostic clinical preservation", "authored exact notes", "authored notes"], interactive=False)
                repeat_status = gr.Markdown()
                repeat_refresh.click(task_repeat_evidence, [], [repeat_table, repeat_status], api_name="task_repeat_evidence")
            with gr.Accordion("Task-piece instructor prompts: expected versus generated", open=False):
                task_model = gr.Dropdown(list(ARCHITECTURES), value="transformer_2", label="Task-piece architecture")
                task_button = gr.Button("Read task-piece prompt outputs")
                task_outputs = gr.Markdown()
                task_button.click(task_acceptance, task_model, task_outputs, api_name="task_acceptance")
            with gr.Accordion("Current Mitra: complete held-out partitions", open=False):
                task_final_button = gr.Button("Load current held-out results")
                task_final_table = gr.Dataframe(headers=["architecture", "partition", "notes", "recall", "precision", "exact match", "clinical preservation", "note leakage"], interactive=False)
                task_final_status = gr.Markdown()
                task_final_button.click(task_final_partitions, [], [task_final_table, task_final_status], api_name="task_final_partitions")
            with gr.Accordion("Selected demo: separate held-out evidence", open=False):
                demo_final_button = gr.Button("Load selected-demo held-out results")
                demo_final_table = gr.Dataframe(headers=["partition", "notes", "recall", "precision", "exact match", "clinical preservation", "note leakage"], interactive=False)
                demo_final_status = gr.Markdown()
                demo_final_button.click(demo_final_partitions, [], [demo_final_table, demo_final_status], api_name="demo_final_partitions")
            with gr.Accordion("Earlier character studies and their findings", open=False):
                gr.Markdown("## Version 1 — flat-field study\nSaved benchmark results: generated-note metrics use the declared subset; validation token metrics use teacher forcing. Components select separately trained models. Read support counts and seed limitations.")
                table = gr.Dataframe(headers=columns, interactive=False)
                curves = gr.Plot()
                with gr.Accordion("Raw saved head-ablation evidence", open=False):
                    head_results = gr.JSON()
                refresh = gr.Button("Load saved evidence")
                refresh.click(saved_evidence, [], [table, curves, head_results], api_name="evidence")
                findings_button = gr.Button("Read reviewed findings")
                findings = gr.Markdown()
                findings_button.click(reviewed_findings, [], findings, api_name="findings")
                with gr.Accordion("Generalization across complete held-out partitions", open=False):
                    final_button = gr.Button("Load held-out partition results")
                    final_table = gr.Dataframe(headers=["architecture", "partition", "notes", "recall", "precision", "exact match", "clinical preservation", "note leakage"], interactive=False)
                    final_status = gr.Markdown()
                    final_button.click(final_partition_evidence, [], [final_table, final_status], api_name="final_partitions")
                with gr.Accordion("Checkpoint selection: loss versus task outputs", open=False):
                    selection_button = gr.Button("Load matched checkpoint comparison")
                    selection_evidence = gr.Markdown()
                    selection_button.click(checkpoint_selection_evidence, [], selection_evidence, api_name="checkpoint_selection")
                gr.Markdown("## Version 2 — instructor sentence forms\nThe original dataset missed sentence forms such as VISITED/ON and LIVES. This corrected series keeps its own data, checkpoints and measurements.")
                assignment_table = gr.Dataframe(headers=["architecture", "selected update", "total updates", "parameters", "val loss", "val token accuracy", "diagnostic exact", "diagnostic recall", "diagnostic clinical preservation", "authored exact notes", "authored notes"], interactive=False)
                assignment_curves = gr.Plot()
                assignment_status = gr.Markdown()
                assignment_refresh = gr.Button("Load assignment-format results")
                assignment_refresh.click(assignment_evidence, [], [assignment_table, assignment_curves, assignment_status], api_name="assignment_evidence")
                with gr.Accordion("Expected labels versus actual instructor-prompt generations", open=False):
                    acceptance_model = gr.Dropdown(list(ARCHITECTURES), value="transformer_2", label="Version-2 architecture")
                    acceptance_button = gr.Button("Read saved instructor-prompt outputs")
                    acceptance_outputs = gr.Markdown()
                    acceptance_button.click(assignment_acceptance, acceptance_model, acceptance_outputs, api_name="assignment_acceptance")
        with gr.Accordion("Trace a token through the model", open=False):
            gr.Markdown("Text → the selected tokenizer's pieces → integer IDs → token + position embeddings → custom LayerNorm → explicit Q/K/V and causal attention → residual → FFN and residual → second block → final normalization → vocabulary logits → greedy/sample choice → append → repeat until EOS. Architecture variants omit specified parts.")
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--assignment-root", type=Path, default=ASSIGNMENT_ROOT, help="Separate downloaded cloud study directory; original local files stay intact")
    parser.add_argument("--task-root", type=Path, default=TASK_ROOT, help="Directory containing the fixed-task-piece study")
    args = parser.parse_args()
    ASSIGNMENT_ROOT = args.assignment_root.resolve()
    TASK_ROOT = args.task_root.resolve()
    torch.set_num_threads(2)
    build_app().queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False)
