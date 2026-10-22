# What did the heads contribute?

Author self-review, October 18, 2026. This evidence belongs to the **final corpus, two-block model, training seed42, update4,000**. It does not describe another seed's learned heads. The model is already trained; these interventions do not retrain it.

**Question.** Does the trained model depend equally on all eight heads for these predictions?

**Change.** Disable one head by setting its context vector to zero before concatenation and the output projection. Keep weights, input notes, greedy decoding and the other heads unchanged. Evaluate the same eight category-selected validation notes. Indexes start at zero: L0H0 is the first block's first head.

**Observation.** Baseline:8/8 exact notes,14/14 correctly typed identifier replacements,6/6 preserved clinical associations. Conditional next-token loss is approximately0.0002 per scored output token. The small selected subset is easier than the entire validation set and the authored suite.

| Disabled head | Correct identifiers /14 | Exact notes /8 | Conditional loss |
| --- | ---: | ---: | ---: |
| None | 14 | 8 | 0.0002 |
| L0H0 | 4 | 1 | 1.1057 |
| L0H1 | 10 | 6 | 0.3461 |
| L0H2 | 12 | 5 | 0.3369 |
| L0H3 | 14 | 7 | 0.0268 |
| L1H0 | 13 | 7 | 0.0637 |
| L1H1 | 14 | 7 | 0.0193 |
| L1H2 | 13 | 6 | 0.1524 |
| L1H3 | 14 | 7 | 0.0995 |

**Interpretation.** Removing L0H0 has the largest recall drop on this subset. We can say the model depends strongly on that head here. L1H3 preserves all14 identifier replacements but lowers complete-note accuracy and raises conditional loss: importance depends on the outcome being measured. A head is not universally “good” or “bad.”

**Limit.** Eight notes are shared across interventions. Category supports are PATIENT7, MRN2, and one each for DOCTOR, DATE, ADDRESS, DOB and PHONE. L0H0's removal also harms several categories; this does not establish a dedicated MRN head. Heatmaps show routing for a supplied prefix and one next-token query. They do not reveal a complete semantic explanation. No significance or universal superiority claim follows from this descriptive experiment.

Two pairs, heads0+1 in each block, were declared before these final-model interventions. Removing the block0 pair leaves1/14 correct replacements; removing the block1 pair leaves12/14. Pair effects differ from adding the individual effects. Nonlinear transformations and changes in generated prefixes can contribute to that difference; compensation is an interpretation to investigate, not a proven unique cause.

The runner verifies that generation after interventions returns to the baseline outputs. Raw reports retain source, expected target, actual generation, EOS, category counts and full-precision loss. See [the raw evidence](../artifacts/mitra_final/results/investigations/enriched_head_measurements.json) and [the declared protocol](../artifacts/mitra_final/results/investigations/protocol.json).

For the demo, use **Inspect heads → task_tokens**. State the checkpoint caption, read the denominator, show one conditional attention row, then compare one live head intervention with the saved measurements. A new live input has no gold label, so its output alone has no precision or recall score.
