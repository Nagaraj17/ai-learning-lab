"""Render measured files into a readable report; do not invent interpretations."""
import json
import statistics
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    results = Path("artifacts/results")
    summaries = json.loads((results / "benchmark_summary.json").read_text(encoding="utf-8"))
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    lines = ["# Tiny Mitra: measured results", "", "## Version 1: original flat-field study", "", "Local CPU, 1,000 optimizer updates per architecture, training seed 42. Generated-note metrics below use 12 validation diagnostics; teacher-forced validation uses all 100 validation examples. Minimum-validation-loss checkpoints are selected. The original dataset omits some instructor sentence forms; the separate version-2 study corrects coverage. These are synthetic educational measurements, not production de-identification claims.", "", "| Architecture | Validation loss | Token accuracy | Redaction recall | Exact notes | Clinical preservation |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    def value(number): return "N/A" if number is None else f"{number:.3f}"
    for row in summaries:
        lines.append("| " + row["architecture"] + " | " + " | ".join(value(row.get(key)) for key in ("validation_loss", "validation_token_accuracy", "recall", "exact_match", "clinical_preservation")) + " |")
    lines += ["", "![Token-weighted validation curves](validation_curves.png)", "", "## Original-study seed repeats", ""]
    repeats_path = results / "repeats_summary.json"
    repeated = json.loads(repeats_path.read_text(encoding="utf-8")) if repeats_path.exists() else []
    lines += [f"{len(repeated)} of 8 additional runs saved: four architectures, seeds 7 and 123. Seed 42 is the primary study. All use the same 12 validation diagnostics; these repeats vary both weight initialization and minibatch order, not new datasets or independent samples of clinical practice.", "", "| Architecture | Seeds saved | Mean recall [min, max] | Mean exact match [min, max] | Mean clinical preservation [min, max] |", "| --- | --- | --- | --- | --- |"]
    for architecture in ("embedding_only", "single_attention", "multi_attention", "transformer_2"):
        selected = [row for row in summaries + repeated if row["architecture"] == architecture]
        cells = []
        for key in ("recall", "exact_match", "clinical_preservation"):
            values = [row[key] for row in selected if row.get(key) is not None]
            cells.append(f"{statistics.mean(values):.3f} [{min(values):.3f}, {max(values):.3f}]" if values else "N/A")
        lines.append("| " + architecture + " | " + ", ".join(str(row["seed"]) for row in selected) + " | " + " | ".join(cells) + " |")
    lines += ["", "Ranges describe observed seed variation. Three seeds and a small shared diagnostic set do not establish statistical significance or universal architecture rankings.", "", "## Original-study frozen final partitions", ""]
    final_path = results / "final_test/summary.json"
    final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else []
    lines += [f"{len(final)} of 32 architecture/partition rows saved. Each completed row evaluates all 50 notes. Seed-42 checkpoints were frozen before release; these outcomes must not tune the reported models. Known/test conditions describe declared entity/layout pools. Controls share layouts and dates may recur.", "", "| Architecture | Partition | Exact notes | Correct identifiers | Preserved clinical fields | Notes with exact-surface leakage |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in final:
        lines.append(f"| {row['architecture']} | {row['split']} | {row['exact']}/{row['examples']} | {row['tp']}/{row['identifiers']} | {row['clinical_correct']}/{row['clinical_support']} | {row['leaked_note']}/{row['examples']} |")
    lines += ["", "## Version 2: instructor sentence forms", "", "Separate dataset, sentence-anchor evaluator, context256 and registered 2,000-update budget. The original study uses context512 and 1,000 updates. Compare architectures within a series; cross-series differences confound data, context, budget and metrics.", ""]
    assignment_path = Path("artifacts/assignment_v2/results/summary.json")
    assignment = json.loads(assignment_path.read_text(encoding="utf-8")) if assignment_path.exists() else []
    lines += [f"{len(assignment)} of 8 runs have saved development generations. The 24 validation diagnostics and 11 authored instructor/clinical prompts are development evidence, not final-test results.", "", "| Architecture | Selected update | Validation loss | Diagnostic recall | Diagnostic exact | Diagnostic clinical preservation | Authored exact notes |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in assignment:
        cells = [value(row.get(key)) for key in ("validation_loss", "diagnostic_recall", "diagnostic_exact", "diagnostic_clinical_preservation")]
        lines.append(f"| {row['architecture']} | {row['selected_step']} | " + " | ".join(cells) + f" | {row['acceptance_correct_notes']}/{row['acceptance_notes']} |")
    lines += ["", "## Fixed task pieces with character fallback", "", "Same assignment-format source strings and split identities, with a declared fixed vocabulary of task pieces plus ASCII character spelling. This series compares the final 2,000-update checkpoint for every architecture. Minimum-loss checkpoints remain available as separate diagnostics. Vocabulary/sequence lengths, parameter counts and older checkpoint-selection rules differ: compare complete-note/clinical outcomes with their scope, not cross-tokenizer loss/perplexity.", ""]
    task_path = Path("artifacts/task_tokens/results/summary.json")
    task = json.loads(task_path.read_text(encoding="utf-8")) if task_path.exists() else []
    lines += [f"{len(task)} of 8 runs have saved development generations. The separate demo gate requires the full model to produce all 11 authored prompts exactly with valid EOS; it does not establish held-out generalization.", "", "| Architecture | Fixed update | Parameters | Diagnostic exact | Diagnostic recall | Diagnostic clinical preservation | Authored exact notes |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in task:
        cells = [value(row.get(key)) for key in ("diagnostic_exact", "diagnostic_recall", "diagnostic_clinical_preservation")]
        lines.append(f"| {row['architecture']} | {row['selected_step']} | {row['parameters']} | " + " | ".join(cells) + f" | {row['acceptance_correct_notes']}/{row['acceptance_notes']} |")
    task_audit = Path("artifacts/task_tokens/results/audit.json")
    if task_audit.exists():
        audit = json.loads(task_audit.read_text(encoding="utf-8"))
        lines += ["", f"Authored demo gate: {'passed' if audit['demo_ready'] else 'not passed'} ({audit['full_model_authored_exact']}/11 exact notes). Engineering checks and model-quality gates are separate."]
    lines += ["", "## Interpretation and raw evidence", "", "Read hypothesis_reviews.json for question, change, observation, verdict, interpretation and limitation. Read per-run raw JSON for exact IDs, expected/generated text, counts, EOS and timings. Original head pairs were selected using validation diagnostics and are exploratory. Attention inspection is conditional on a supplied output prefix. No semantic head role follows automatically from a heatmap or intervention. A lower teacher-forced loss alone does not establish correct free generation.", "", "## Open limits", "", "Small synthetic grammar and diagnostics; single seed for component ablations and version 2; exact-surface leakage can miss partial/reformatted identifiers; shared control templates; actual cloud execution, cross-device verification and learner teach-back pending. No real clinical data was used."]
    figure, axis = plt.subplots(figsize=(10, 5))
    for directory in Path("artifacts/checkpoints").glob("*_seed42"):
        if directory.name.replace("_seed42", "") not in {row["architecture"] for row in summaries}: continue
        path = directory / "history.json"
        if path.exists():
            history = json.loads(path.read_text(encoding="utf-8"))
            axis.plot([row["step"] for row in history], [row["validation"]["loss"] for row in history], label=directory.name.replace("_seed42", ""))
    axis.set(xlabel="Optimizer updates", ylabel="Output-token-weighted validation loss")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(reports / "validation_curves.png", dpi=160)
    plt.close(figure)
    final_root = Path("artifacts/mitra_final")
    final_path = final_root / "results/summary.json"
    final_models = json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else []
    current = ["# Tiny Mitra: final submission evidence", "", "One frozen corpus of 1,000 synthetic notes: 700 training, 100 validation and four final partitions of 50. Every architecture starts from random weights and uses its registered final 4,000th update. Earlier models do not initialize these weights. Vocabulary146, width64, FFN192, context256; head count/components follow the registry. The minimum-loss checkpoint is kept separately and does not replace the final-update comparison.", "", f"{len(final_models)} of 8 model runs have saved development generations. Passing engineering checks is separate from the authored/clinical quality gates and held-out generalization.", "", "| Architecture | Update | Parameters | Diagnostic exact notes | Correct typed replacements | Preserved clinical associations | Authored exact notes |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    diagnostic_ids = []
    for row in final_models:
        raw = json.loads((final_root / f"results/{row['architecture']}_validation.json").read_text(encoding="utf-8"))["metrics"]
        diagnostic_ids = raw["example_ids"]
        current.append(f"| {row['architecture']} | {row['selected_step']} | {row['parameters']} | {raw['exact']}/{raw['examples']} | {raw['tp']}/{raw['identifiers']} | {raw['clinical_correct']}/{raw['clinical_support']} | {row['acceptance_correct_notes']}/{row['acceptance_notes']} |")
    if diagnostic_ids:
        data = json.loads((final_root / "dataset/examples.json").read_text(encoding="utf-8"))
        coverage = Counter(row["entity_pool"] + "/" + row["template_pool"] for row in data if row["id"] in diagnostic_ids)
        current += ["", "The 24 diagnostic notes are category-selected validation examples. Entity/layout pool coverage: " + ", ".join(f"{key}={count}" for key, count in coverage.items()) + ". Teacher-forced validation uses all 100 notes; complete final partitions are reported separately."]
    for name, title in (("audit.json", "Authored demo gate"), ("clinical_acceptance.json", "Additional clinical preservation checks")):
        path = final_root / "results" / name
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            passed = result.get("demo_ready", result.get("passed"))
            current += ["", f"{title}: {'passed' if passed else 'not passed'}. See {name} for actual outputs/counts and scope."]
    final_partitions = final_root / "results/final_test/summary.json"
    repeat_path = final_root / "results/repeats_summary.json"
    repeat_rows = json.loads(repeat_path.read_text(encoding="utf-8")) if repeat_path.exists() else []
    current += ["", "### Training-seed sensitivity", "", f"{len(repeat_rows)}/8 additional runs complete: four architectures at seeds7/123. Seeds vary both initial weights and minibatch order. These use the same24 validation diagnostics and11 authored notes; descriptive differences do not establish statistical significance. Component ablations beyond these four retain one seed.", "", "| Architecture | Seed | Diagnostic exact notes | Correct identifiers | Preserved clinical associations | Authored exact notes |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in repeat_rows:
        raw = json.loads((final_root / f"results/{row['architecture']}_seed{row['seed']}_validation.json").read_text(encoding="utf-8"))["metrics"]
        current.append(f"| {row['architecture']} | {row['seed']} | {raw['exact']}/{raw['examples']} | {raw['tp']}/{raw['identifiers']} | {raw['clinical_correct']}/{raw['clinical_support']} | {row['acceptance_correct_notes']}/{row['acceptance_notes']} |")
    demo_path = reports / "DEMO_SELECTION.json"
    if demo_path.exists():
        demo = json.loads(demo_path.read_text(encoding="utf-8"))
        current += ["", "Development-only demo selection: " + (demo["selected"]["model_id"] if demo["passed"] else "no passing candidate") + ". The primary seed42 architecture comparison remains unchanged. See DEMO_SELECTION_PROTOCOL.json for timing disclosure, priority and gate; final-test outcomes do not choose this candidate."]
    partitions = json.loads(final_partitions.read_text(encoding="utf-8")) if final_partitions.exists() else []
    current += ["", f"Complete final partitions: {len(partitions)}/32 rows saved. Release requires the separate final protocol review. Fifty notes per row; 200 distinct test notes reused across eight models. Development names are excluded from the final name pool. Controls share layouts and dates/doctor/address policies retain their declared limitations.", "", "| Architecture | Partition | Exact notes | Correct identifiers | Clinical associations |", "| --- | --- | ---: | ---: | ---: |"]
    for row in partitions: current.append(f"| {row['architecture']} | {row['split']} | {row['exact']}/{row['examples']} | {row['tp']}/{row['identifiers']} | {row['clinical_correct']}/{row['clinical_support']} |")
    demo_final_path = final_root / "results/demo_final_test/summary.json"
    if demo_final_path.exists():
        demo_final = json.loads(demo_final_path.read_text(encoding="utf-8"))
        current += ["", "### Already development-selected demo: separate final-test evidence", "", f"Model: {demo_final.get('model_id', 'none')}. Status: {demo_final['status']}. Its selection was fixed from development evidence before these outcomes. Same200 test notes as the primary comparison; no claim of additional independent cases.", "", "| Partition | Exact notes | Correct identifiers | Clinical associations |", "| --- | ---: | ---: | ---: |"]
        for row in demo_final["rows"]: current.append(f"| {row['split']} | {row['exact']}/{row['examples']} | {row['tp']}/{row['identifiers']} | {row['clinical_correct']}/{row['clinical_support']} |")
    current += ["", "## Earlier studies â€” separate histories", "", "The following results preserve earlier datasets, tokenizers, budgets and selection rules. They do not establish robustness of the final model and must not be relabelled as its metrics.", ""]
    earlier_index = next(i for i, line in enumerate(current) if line.startswith("## Earlier studies"))
    current[earlier_index:earlier_index] = ["## Final-corpus interpretation", "", "Read [FINAL_FINDINGS.md](FINAL_FINDINGS.md) for all nine reviewed questions, observations, verdicts and limits. These are author self-reviews, not independent sign-off or statistical proof. The final study is a descriptive follow-up after earlier development studies. The five exact notes on each new-layout partition for the full models are the five no-identifier controls; this is not successful redaction of novel layouts.", "", "[Recomputed test loss and next-token accuracy](../artifacts/mitra_final/results/final_test/token_metrics_audit.json) covers all32 primary model/partition rows. Teacher forcing supplies the correct output prefix and scores output/EOS tokens, so it differs from free-generation exact match. Histories retain last-minibatch training loss and full100-note validation metrics.", ""]
    lines = current + lines[1:]
    (reports / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Rendered measured report and validation curves")


if __name__ == "__main__":
    main()
