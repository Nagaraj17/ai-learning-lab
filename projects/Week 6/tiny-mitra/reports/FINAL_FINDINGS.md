# Final-corpus findings

Final frozen single-corpus study. Questions follow the earlier plan; final-corpus interpretations are descriptive follow-up, not statistical hypothesis proof. No test-driven retuning.

author self-review; not independent sign-off. Learner teach-back remains open.

## H1: Does attention enable source-dependent next-token behavior?

**Change:** Change the one atomic source token DrugA/DrugB; keep the supplied output prefix and final query position fixed across four checkpoints.

**Observation:** Embedding-only gives exactly identical distributions and selects [MRN] twice. Single-head and full seed42 select DrugA then DrugB and freely generate both paired notes correctly. Four-head selects Chemotherapy then DrugB and fails the first pair.

**Verdict:** Local context dependence is demonstrated by the single-head and full checkpoints; useful behavior is not automatic for every attention model.

**Interpretation:** Context mixing makes earlier source information available. The trained checkpoint must still learn to use it correctly.

**Limitations:** Two authored inputs and a supplied prefix, plus their separate free generations; no claim of broad transfer or a semantic head role.

Evidence: [architecture_context.json](../artifacts/mitra_final/results/investigations/architecture_context.json).

## H2: Do four bare heads improve behavior over one?

**Change:** Keep width64 and51858 parameters; compare one versus four bare heads at seeds42/7/123.

**Observation:** Single-head exact counts are22,22,18/24; four-head20,24,22/24. Mean exact is86.1% versus91.7%; mean recall95.4% versus96.3%. Authored notes instead average10/11 versus8.67/11.

**Verdict:** Mixed, metric-dependent evidence; four heads are not consistently better in paired seeds or authored examples.

**Interpretation:** Projection diversity may help some diagnostics but does not guarantee better complete task behavior.

**Limitations:** One frozen synthetic corpus and fixed update4000; diagnostics are24 familiar-condition validation notes. Equal width/updates do not mean equal parameters or compute. Three seeds vary both initialization and minibatch order on shared notes; no p-value or universal superiority claim.

Evidence: [summary.json](../artifacts/mitra_final/results/summary.json), [repeats_summary.json](../artifacts/mitra_final/results/repeats_summary.json), [descriptive_comparison.json](../artifacts/mitra_final/results/descriptive_comparison.json).

## H3: Does a second block improve generated quality?

**Change:** Train one full block versus two;77074 versus118802 parameters.

**Observation:** One block: 23/24 exact notes, 35/36 identifiers, 24/24 clinical associations; two: 23/24 exact notes, 36/36 identifiers, 23/24 clinical associations. Authored11/11 versus10/11. Familiar-layout final exact48/50 and50/50 across names for one block,47/50 and50/50 for two. Both score5/50 on each new-layout partition.

**Verdict:** No consistent advantage for the extra block under this budget.

**Interpretation:** Additional capacity and lower loss need not improve generated complete notes.

**Limitations:** One frozen synthetic corpus and fixed update4000; diagnostics are24 familiar-condition validation notes. Equal width/updates do not mean equal parameters or compute. One primary seed for the depth comparison; no component-specific learning-rate tuning.

Evidence: [summary.json](../artifacts/mitra_final/results/summary.json), [summary.json](../artifacts/mitra_final/results/final_test/summary.json).

## H4: Does LayerNorm help optimization or output quality?

**Change:** Remove all block norms and final norm; retain the declared optimizer and clipping policy.

**Observation:** Full: 23/24 exact notes, 36/36 identifiers, 23/24 clinical associations; without norm: 23/24 exact notes, 36/36 identifiers, 23/24 clinical associations. Maximum logged pre-clip norm 1.378 versus 2.693. Largest logged positive validation-loss jump0.146 versus0.368. Both finish4000 updates; authored10/11 versus9/11.

**Verdict:** Some setup-specific optimization/scale evidence; diagnostic generated-quality counts tie.

**Interpretation:** Normalization changes feature scale and optimization behavior, but these runs do not show that it is required for all learning.

**Limitations:** One frozen synthetic corpus and fixed update4000; diagnostics are24 familiar-condition validation notes. Equal width/updates do not mean equal parameters or compute. Gradient statistics sample40 logging intervals, not every update; all runs use clipping. One ablation seed; fewer parameters after norm removal.

Evidence: [descriptive_comparison.json](../artifacts/mitra_final/results/descriptive_comparison.json), [summary.json](../artifacts/mitra_final/results/summary.json).

## H5: Do residual paths improve learning here?

**Change:** Train without the attention and FFN residual additions; same118802 parameters.

**Observation:** Full: 23/24 exact notes, 36/36 identifiers, 23/24 clinical associations; without residuals: 24/24 exact notes, 36/36 identifiers, 24/24 clinical associations. Without residuals gets50/50 on both familiar-layout final partitions versus47/50 and50/50 for full. Both get5/50 on both new-layout conditions. Max logged pre-clip norms1.378 versus3.109; both finish.

**Verdict:** A benefit to generated quality is not supported here; the no-residual run performs better on some measured outcomes.

**Interpretation:** Residual paths provide a direct signal route, but their theoretical purpose does not force superiority in this small fixed-budget experiment.

**Limitations:** One frozen synthetic corpus and fixed update4000; diagnostics are24 familiar-condition validation notes. Equal width/updates do not mean equal parameters or compute. One seed and shallow models; larger gradient maxima alone are not divergence or proof of easier learning.

Evidence: [summary.json](../artifacts/mitra_final/results/summary.json), [descriptive_comparison.json](../artifacts/mitra_final/results/descriptive_comparison.json), [summary.json](../artifacts/mitra_final/results/final_test/summary.json).

## H6: Does the FFN improve the transformation?

**Change:** Remove both FFNs and their unused preceding norms;118802 versus68882 parameters.

**Observation:** Full: 23/24 exact notes, 36/36 identifiers, 23/24 clinical associations; without FFN: 23/24 exact notes, 35/36 identifiers, 24/24 clinical associations. Authored10/11 versus11/11. Familiar-layout final exact47/50,50/50 versus48/50,50/50. Both get5/50 on both new-layout conditions.

**Verdict:** Mixed; this experiment does not establish an FFN advantage in complete-note behavior.

**Interpretation:** The FFN transforms features at each position, while attention moves information across positions. Removing it changes capacity and optimization as well as this mechanism.

**Limitations:** One frozen synthetic corpus and fixed update4000; diagnostics are24 familiar-condition validation notes. Equal width/updates do not mean equal parameters or compute. One primary seed; no claim about deeper models or broader language tasks.

Evidence: [summary.json](../artifacts/mitra_final/results/summary.json), [summary.json](../artifacts/mitra_final/results/final_test/summary.json).

## H7: Do individual heads make different contributions?

**Change:** Zero each head context before merge/output projection, then test two fixed0+1 pairs; leave trained weights unchanged.

**Observation:** Baseline14/14 identifiers and8/8 notes, loss0.000174. Removing L0H0 yields4/14 and1/8, loss1.1057. L0H3 and L1H1 retain14/14 recall but fall to7/8 exact notes. Paired layer0 removal gives1/14 and1/8; layer1 gives12/14 and4/8.

**Verdict:** Different checkpoint dependencies are observed; recall alone misses degradation. Pair effects are non-additive.

**Interpretation:** Use the attention row, controlled source edits and category metrics together. Neither pair non-additivity nor a heatmap uniquely identifies compensation or semantic specialization.

**Limitations:** Seed42 only, eight validation notes: PATIENT7,MRN2 and other categories1 each. Demo seed7 heads were not investigated. Both pair choices were fixed before these interventions.

Evidence: [protocol.json](../artifacts/mitra_final/results/investigations/protocol.json), [enriched_head_measurements.json](../artifacts/mitra_final/results/investigations/enriched_head_measurements.json), [audit.json](../artifacts/mitra_final/results/investigations/audit.json).

## H8: Does redaction transfer to new names and layouts?

**Change:** Evaluate all four50-note final partitions after freezing protocol and demo selection; never select weights from these outcomes.

**Observation:** Primary42 full exact:47,5,50,5/50 (known/known,known/new,new/known,new/new). Selected demo7:49,5,43,5/50. Demo7 identifiers:88/89,1/128,68/86,0/142. The five exact new-layout notes are the five no-identifier controls, not successful redaction of novel layouts.

**Verdict:** Substantial familiar-layout behavior and partial new-name transfer; new-layout redaction fails severely.

**Interpretation:** Supported spelling and successful authored prompts do not establish compositional template generalization. Zero exact-surface leakage can coexist with omissions and incorrect transformations.

**Limitations:** 200 distinct test notes reused across eight primary models and the separately selected demo; no independent clinical population. Novelty applies to declared pools; some control/identifier patterns recur. Both extra clinical demo checks occur in training. Do not tune this frozen result after seeing tests.

Evidence: [summary.json](../artifacts/mitra_final/results/final_test/summary.json), [summary.json](../artifacts/mitra_final/results/demo_final_test/summary.json), [demo_seed7_development_audit.json](../artifacts/mitra_final/results/demo_seed7_development_audit.json).

## H9: Does sampling temperature change output reliability?

**Change:** Sample eight notes at0.1,0.7,1.2 with matched per-note random seeds; greedy is separate.

**Observation:** All three sampled temperatures produce8/8 exact notes,14/14 identifiers and valid EOS on all8 notes. No degradation is observed on this sample. Separately, OOD source 'PATIENT Kai DIAGNOSIS UNKNOWN' generates 'PATIENT [NAME] DIAGNOSIS DIABETES', with selected DIABETES probability about0.811.

**Verdict:** A temperature-quality effect is not detected here. The separate OOD probe demonstrates unsupported substitution, not a temperature effect.

**Interpretation:** Temperature rescales logits; already concentrated distributions can give identical sampled outputs. High next-token probability measures the learned continuation preference rather than factual support or correctness.

**Limitations:** Eight notes and one sampling seed per note, not a robust distribution estimate. OOD probes are qualitative and have no invented aggregate gold accuracy.

Evidence: [temperature_0.1.json](../artifacts/mitra_final/results/investigations/temperature_0.1.json), [temperature_0.7.json](../artifacts/mitra_final/results/investigations/temperature_0.7.json), [temperature_1.2.json](../artifacts/mitra_final/results/investigations/temperature_1.2.json), [ood.json](../artifacts/mitra_final/results/investigations/ood.json).
