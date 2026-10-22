"""Assemble author-reviewed interpretations from audited, immutable evidence.

The questions follow the original plan. These final-corpus observations are a
separate application after earlier development studies, not a preregistered
independent replication or a statistical significance test.
"""
import json
from pathlib import Path
from train import atomic_json


def main():
    root = Path("artifacts/mitra_final/results")
    read = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    assert read("audit.json")["engineering_audit_passed"]
    assert read("repeats_audit.json")["passed"]
    assert read("final_test/audit.json")["passed"]
    assert read("final_test/token_metrics_audit.json")["passed"]
    comparison = read("descriptive_comparison.json")
    diag = {row["architecture"]: row for row in comparison["training_diagnostics"]}
    metrics = {row["architecture"]: read(row["architecture"] + "_validation.json")["metrics"] for row in read("summary.json")}
    def counts(name):
        row = metrics[name]
        return f"{row['exact']}/24 exact notes, {row['tp']}/36 identifiers, {row['clinical_correct']}/24 clinical associations"
    shared = "One frozen synthetic corpus and fixed update4000; diagnostics are24 familiar-condition validation notes. Equal width/updates do not mean equal parameters or compute."
    rows = []
    def add(number, question, change, observation, verdict, interpretation, limits, evidence):
        rows.append(dict(hypothesis_id=f"H{number}", question=question, change=change, observation=observation, verdict=verdict, interpretation=interpretation, limitations=limits, evidence=[str(root / path).replace('\\', '/') for path in evidence]))
    add(1, "Does attention enable source-dependent next-token behavior?",
        "Change the one atomic source token DrugA/DrugB; keep the supplied output prefix and final query position fixed across four checkpoints.",
        "Embedding-only gives exactly identical distributions and selects [MRN] twice. Single-head and full seed42 select DrugA then DrugB and freely generate both paired notes correctly. Four-head selects Chemotherapy then DrugB and fails the first pair.",
        "Local context dependence is demonstrated by the single-head and full checkpoints; useful behavior is not automatic for every attention model.",
        "Context mixing makes earlier source information available. The trained checkpoint must still learn to use it correctly.",
        "Two authored inputs and a supplied prefix, plus their separate free generations; no claim of broad transfer or a semantic head role.", ["investigations/architecture_context.json"])
    add(2, "Do four bare heads improve behavior over one?", "Keep width64 and51858 parameters; compare one versus four bare heads at seeds42/7/123.",
        "Single-head exact counts are22,22,18/24; four-head20,24,22/24. Mean exact is86.1% versus91.7%; mean recall95.4% versus96.3%. Authored notes instead average10/11 versus8.67/11.",
        "Mixed, metric-dependent evidence; four heads are not consistently better in paired seeds or authored examples.",
        "Projection diversity may help some diagnostics but does not guarantee better complete task behavior.", shared + " Three seeds vary both initialization and minibatch order on shared notes; no p-value or universal superiority claim.", ["summary.json", "repeats_summary.json", "descriptive_comparison.json"])
    add(3, "Does a second block improve generated quality?", "Train one full block versus two;77074 versus118802 parameters.",
        "One block: " + counts("transformer_1") + "; two: " + counts("transformer_2") + ". Authored11/11 versus10/11. Familiar-layout final exact48/50 and50/50 across names for one block,47/50 and50/50 for two. Both score5/50 on each new-layout partition.",
        "No consistent advantage for the extra block under this budget.", "Additional capacity and lower loss need not improve generated complete notes.", shared + " One primary seed for the depth comparison; no component-specific learning-rate tuning.", ["summary.json", "final_test/summary.json"])
    add(4, "Does LayerNorm help optimization or output quality?", "Remove all block norms and final norm; retain the declared optimizer and clipping policy.",
        "Full: " + counts("transformer_2") + "; without norm: " + counts("no_layernorm") + f". Maximum logged pre-clip norm {diag['transformer_2']['maximum_logged_gradient_norm_before_clip']:.3f} versus {diag['no_layernorm']['maximum_logged_gradient_norm_before_clip']:.3f}. Largest logged positive validation-loss jump0.146 versus0.368. Both finish4000 updates; authored10/11 versus9/11.",
        "Some setup-specific optimization/scale evidence; diagnostic generated-quality counts tie.",
        "Normalization changes feature scale and optimization behavior, but these runs do not show that it is required for all learning.", shared + " Gradient statistics sample40 logging intervals, not every update; all runs use clipping. One ablation seed; fewer parameters after norm removal.", ["descriptive_comparison.json", "summary.json"])
    add(5, "Do residual paths improve learning here?", "Train without the attention and FFN residual additions; same118802 parameters.",
        "Full: " + counts("transformer_2") + "; without residuals: " + counts("no_residual") + ". Without residuals gets50/50 on both familiar-layout final partitions versus47/50 and50/50 for full. Both get5/50 on both new-layout conditions. Max logged pre-clip norms1.378 versus3.109; both finish.",
        "A benefit to generated quality is not supported here; the no-residual run performs better on some measured outcomes.",
        "Residual paths provide a direct signal route, but their theoretical purpose does not force superiority in this small fixed-budget experiment.", shared + " One seed and shallow models; larger gradient maxima alone are not divergence or proof of easier learning.", ["summary.json", "descriptive_comparison.json", "final_test/summary.json"])
    add(6, "Does the FFN improve the transformation?", "Remove both FFNs and their unused preceding norms;118802 versus68882 parameters.",
        "Full: " + counts("transformer_2") + "; without FFN: " + counts("no_ffn") + ". Authored10/11 versus11/11. Familiar-layout final exact47/50,50/50 versus48/50,50/50. Both get5/50 on both new-layout conditions.",
        "Mixed; this experiment does not establish an FFN advantage in complete-note behavior.",
        "The FFN transforms features at each position, while attention moves information across positions. Removing it changes capacity and optimization as well as this mechanism.", shared + " One primary seed; no claim about deeper models or broader language tasks.", ["summary.json", "final_test/summary.json"])
    add(7, "Do individual heads make different contributions?", "Zero each head context before merge/output projection, then test two fixed0+1 pairs; leave trained weights unchanged.",
        "Baseline14/14 identifiers and8/8 notes, loss0.000174. Removing L0H0 yields4/14 and1/8, loss1.1057. L0H3 and L1H1 retain14/14 recall but fall to7/8 exact notes. Paired layer0 removal gives1/14 and1/8; layer1 gives12/14 and4/8.",
        "Different checkpoint dependencies are observed; recall alone misses degradation. Pair effects are non-additive.",
        "Use the attention row, controlled source edits and category metrics together. Neither pair non-additivity nor a heatmap uniquely identifies compensation or semantic specialization.",
        "Seed42 only, eight validation notes: PATIENT7,MRN2 and other categories1 each. Demo seed7 heads were not investigated. Both pair choices were fixed before these interventions.", ["investigations/protocol.json", "investigations/enriched_head_measurements.json", "investigations/audit.json"])
    add(8, "Does redaction transfer to new names and layouts?", "Evaluate all four50-note final partitions after freezing protocol and demo selection; never select weights from these outcomes.",
        "Primary42 full exact:47,5,50,5/50 (known/known,known/new,new/known,new/new). Selected demo7:49,5,43,5/50. Demo7 identifiers:88/89,1/128,68/86,0/142. The five exact new-layout notes are the five no-identifier controls, not successful redaction of novel layouts.",
        "Substantial familiar-layout behavior and partial new-name transfer; new-layout redaction fails severely.",
        "Supported spelling and successful authored prompts do not establish compositional template generalization. Zero exact-surface leakage can coexist with omissions and incorrect transformations.",
        "200 distinct test notes reused across eight primary models and the separately selected demo; no independent clinical population. Novelty applies to declared pools; some control/identifier patterns recur. Both extra clinical demo checks occur in training. Do not tune this frozen result after seeing tests.", ["final_test/summary.json", "demo_final_test/summary.json", "demo_seed7_development_audit.json"])
    add(9, "Does sampling temperature change output reliability?", "Sample eight notes at0.1,0.7,1.2 with matched per-note random seeds; greedy is separate.",
        "All three sampled temperatures produce8/8 exact notes,14/14 identifiers and valid EOS on all8 notes. No degradation is observed on this sample. Separately, OOD source 'PATIENT Kai DIAGNOSIS UNKNOWN' generates 'PATIENT [NAME] DIAGNOSIS DIABETES', with selected DIABETES probability about0.811.",
        "A temperature-quality effect is not detected here. The separate OOD probe demonstrates unsupported substitution, not a temperature effect.",
        "Temperature rescales logits; already concentrated distributions can give identical sampled outputs. High next-token probability measures the learned continuation preference rather than factual support or correctness.",
        "Eight notes and one sampling seed per note, not a robust distribution estimate. OOD probes are qualitative and have no invented aggregate gold accuracy.", ["investigations/temperature_0.1.json", "investigations/temperature_0.7.json", "investigations/temperature_1.2.json", "investigations/ood.json"])
    assert len(rows) == 9
    for row in rows:
        assert all(Path(path).exists() for path in row["evidence"])
    report = {"status": "passed", "review_type": "author self-review; not independent sign-off", "date": "2026-10-18", "scope": "Final frozen single-corpus study. Questions follow the earlier plan; final-corpus interpretations are descriptive follow-up, not statistical hypothesis proof. No test-driven retuning.", "hypotheses": rows}
    atomic_json("reports/final_hypothesis_reviews.json", report)
    lines = ["# Final-corpus findings", "", report["scope"], "", report["review_type"] + ". Learner teach-back remains open."]
    for row in rows:
        lines += ["", "## " + row["hypothesis_id"] + ": " + row["question"], ""]
        for key in ("change", "observation", "verdict", "interpretation", "limitations"):
            lines += [f"**{key.capitalize()}:** {row[key]}", ""]
        lines += ["Evidence: " + ", ".join(f"[{Path(path).name}](../{path})" for path in row["evidence"]) + "."]
    Path("reports/FINAL_FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved nine explicitly scoped final-corpus author reviews")


if __name__ == "__main__":
    main()
