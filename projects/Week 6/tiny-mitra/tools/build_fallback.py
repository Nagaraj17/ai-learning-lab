"""Build an offline HTML snapshot from saved, actual model outputs.

This is a presentation fallback, not live inference. It deliberately shows
expected labels separately and retains failure cases instead of repairing text.
"""
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative, default=None):
    path = ROOT / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def table(headers, rows):
    return "<div class='scroll'><table><thead><tr>" + "".join("<th>" + escape(str(value)) + "</th>" for value in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>" + escape(str(value)) + "</td>" for value in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def prompts(report):
    if not report: return "<p>Saved generations are not available yet.</p>"
    cards = []
    for row in report["outputs"]:
        exact = row["counts"]["exact"]
        cards.append("<details><summary>" + escape(row["id"]) + (" — exact" if exact else " — differs from label") + "</summary><pre>" + escape("Source:    " + row["source"] + "\nExpected:  " + row["expected"] + "\nGenerated: " + row["generated"]) + "</pre><p>Stop: " + escape(row["termination"]) + "</p></details>")
    return "".join(cards)


def main():
    baseline = read("artifacts/results/benchmark_summary.json", [])
    final = read("artifacts/results/final_test/summary.json", [])
    assignment = read("artifacts/assignment_v2/results/summary.json", [])
    task = read("artifacts/task_tokens/results/summary.json", [])
    current = read("artifacts/mitra_final/results/summary.json", [])
    sections = ["<h1>Tiny Mitra — saved demo evidence</h1>", "<p class='notice'>Offline snapshot of actual saved runs. This page does not generate text. Expected labels, measured outputs and study versions remain separate.</p>", "<p>Snapshot: " + datetime.now(timezone.utc).isoformat(timespec="seconds") + ". Synthetic educational notes only. Real cloud execution and learner teach-back remain pending.</p>", "<nav><a href='#pilot'>Teaching pilot</a><a href='#assignment'>Instructor forms</a><a href='#original'>Original study</a><a href='#heldout'>Held-out partitions</a></nav>", "<h2 id='pilot'>Start with the four-note teaching pilot</h2><p>The width32, two-block pilot learned four short training notes. Both controlled Jo/DrugA and Jo/DrugB continuations were observed. This is a mechanism demonstration, not the 1,000-example benchmark.</p>"]
    for name, caption in (("demo-pilot.png", "Actual DrugA teaching-pilot generation"), ("demo-pilot-drugb.png", "Actual DrugB teaching-pilot generation")):
        if (ROOT / "reports" / name).exists(): sections.append(f"<figure><img src='{name}' alt='{caption}'><figcaption>{caption}</figcaption></figure>")
    sections += ["<h2 id='current'>Current final submission</h2>", f"<p>{len(current)} of8 models have saved development outputs. One corpus of1,000 notes; random initial weights;4,000updates per architecture. The eleven authored prompts are development checks, not final-test accuracy.</p>"]
    sections.append(table(("Architecture", "Update", "Authored exact notes", "Diagnostic exact", "Clinical preservation"), [(row["architecture"], row["selected_step"], f"{row['acceptance_correct_notes']}/{row['acceptance_notes']}", row["diagnostic_exact"], row["diagnostic_clinical_preservation"]) for row in current]))
    for row in current:
        sections += ["<h3>Current Mitra: " + escape(row["architecture"]) + "</h3>", prompts(read(f"artifacts/mitra_final/results/{row['architecture']}_acceptance.json"))]
    candidate = read("artifacts/mitra_final/results/transformer_2_seed7_acceptance.json")
    if candidate:
        sections += ["<h2>Separately labelled demo candidate: full model seed7</h2><p>Same final corpus, random initialization and4,000updates. This candidate uses development evidence; the seed42 benchmark and its failures remain above. It is not selected from final-test outcomes.</p>", prompts(candidate)]
        if (ROOT / "reports/demo-final-phone-seeds.png").exists():
            sections.append("<figure><img src='demo-final-phone-seeds.png' alt='Actual seed42 and seed7 phone generations with checkpoint labels'><figcaption>Actual browser comparison; primary failure retained beside passing development repeat.</figcaption></figure>")
    current_final = read("artifacts/mitra_final/results/final_test/summary.json", [])
    sections += ["<h2>Current final-corpus held-out partitions</h2>", f"<p>{len(current_final)}/32 rows saved;50 notes per row,200 distinct notes reused across eight models. This evidence belongs to the final corpus, separate from the earlier character study.</p>", table(("Architecture", "Partition", "Exact notes", "Correct identifiers", "Clinical fields"), [(row["architecture"], row["split"], f"{row['exact']}/{row['examples']}", f"{row['tp']}/{row['identifiers']}", f"{row['clinical_correct']}/{row['clinical_support']}") for row in current_final])]
    sections += ["<h2>Earlier fixed-piece study</h2>", f"<p>{len(task)} of 8 runs have saved development outputs. This earlier study used2,000updates and the narrower input corpus. Its numbers are not current final-model results.</p>"]
    sections.append(table(("Architecture", "Fixed update", "Authored exact notes", "Diagnostic exact", "Clinical preservation"), [(row["architecture"], row["selected_step"], f"{row['acceptance_correct_notes']}/{row['acceptance_notes']}", row["diagnostic_exact"], row["diagnostic_clinical_preservation"]) for row in task]))
    for row in task:
        sections += ["<h3>Task pieces: " + escape(row["architecture"]) + "</h3>", prompts(read(f"artifacts/task_tokens/results/{row['architecture']}_acceptance.json"))]
    sections += ["<h2 id='assignment'>Version 2 — instructor sentence forms, characters</h2>", f"<p>{len(assignment)} of 8 runs have saved development outputs. Separate dataset, context256, 2,000-update protocol; 24 validation diagnostics and 11 authored prompts. These are not final-test scores.</p>"]
    sections.append(table(("Architecture", "Selected update", "Authored exact notes", "Diagnostic exact", "Clinical preservation"), [(row["architecture"], row["selected_step"], f"{row['acceptance_correct_notes']}/{row['acceptance_notes']}", row["diagnostic_exact"], row["diagnostic_clinical_preservation"]) for row in assignment]))
    for row in assignment:
        sections += ["<h3>" + escape(row["architecture"]) + "</h3>", prompts(read(f"artifacts/assignment_v2/results/{row['architecture']}_acceptance.json"))]
    sections += ["<h2 id='original'>Version 1 — original flat-field study</h2><p>Eight seed42 models, 1,000 updates. The free-generation table uses 12 validation diagnostics; teacher-forced validation uses 100 notes. This grammar missed some instructor sentence forms. Its numbers must not be mixed with version 2.</p>"]
    sections.append(table(("Architecture", "Validation loss", "Recall", "Exact match", "Clinical preservation"), [(row["architecture"], *(round(row[key], 3) if row[key] is not None else "N/A" for key in ("validation_loss", "recall", "exact_match", "clinical_preservation"))) for row in baseline]))
    for row in baseline:
        sections += ["<h3>" + escape(row["architecture"]) + " — all 12 saved diagnostics</h3>", prompts(read(f"artifacts/results/{row['architecture']}_seed42_validation.json"))]
    sections += ["<h2 id='heldout'>Original-study frozen final partitions</h2>", f"<p>{len(final)} of 32 rows saved. Each row evaluates all50 notes in one declared entity/layout condition. Checkpoints were frozen before release. Shared control layouts and recurring date values limit novelty claims.</p>"]
    sections.append(table(("Architecture", "Partition", "Exact notes", "Correct identifiers", "Clinical fields"), [(row["architecture"], row["split"], f"{row['exact']}/{row['examples']}", f"{row['tp']}/{row['identifiers']}", f"{row['clinical_correct']}/{row['clinical_support']}") for row in final]))
    sections += ["<h2>How to explain a finding</h2><p>Question → change → observation → interpretation → limit. Attention describes internal routing for a prediction. A head intervention measures dependence at a particular checkpoint; neither automatically establishes a semantic head role. Low exact-surface leakage alone does not establish useful or safe clinical transformation.</p>"]
    document = "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Tiny Mitra saved demo evidence</title><style>body{font:17px/1.55 system-ui,sans-serif;color:#17313e;background:#f6f4ee;max-width:1100px;margin:auto;padding:32px}h1,h2{line-height:1.2}h2{margin-top:48px}a{color:#176b68}nav{display:flex;gap:24px;flex-wrap:wrap;margin:28px 0}.notice{background:#e1eee8;border-left:5px solid #207b70;padding:18px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th{background:#17313e;color:white}td,th{text-align:left;border:1px solid #ddd;padding:10px}details{background:white;border:1px solid #ddd;margin:10px 0;padding:14px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 monospace}img{width:100%;height:auto;border:1px solid #ddd}figure{margin:24px 0}figcaption{font-size:14px;color:#526772}</style><main>" + "\n".join(sections) + "</main></html>"
    (ROOT / "reports/demo_fallback.html").write_text(document, encoding="utf-8")
    print(f"Saved offline fallback: {len(baseline)} original, {len(assignment)} assignment, {len(final)} final-partition rows")


if __name__ == "__main__":
    main()
