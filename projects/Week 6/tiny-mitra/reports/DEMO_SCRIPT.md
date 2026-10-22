# Demonstration and explanation rehearsal

Rehearsal script. Team names, exact speaking duration and timed delivery remain pending. The development-selected live full model is `final_seed7/transformer_2`. The primary architecture comparison and saved head interventions remain seed42. Use the measured report for final held-out outcomes; this script does not certify rehearsal or learner understanding.

## Proposed 45-minute version

| Time | Content | Presenter preparation |
| --- | --- | --- |
| 0–4 min | Business task, synthetic scope, source/expected/generated distinction | Explain why preserving treatment matters as much as replacing a name |
| 4–10 min | Tokenization, unseen names, embeddings and position | Follow an actual token ID and embedding row in the shared notebook |
| 10–20 min | Q/K/V, masks, heads, FFN, residuals, custom LayerNorm and two blocks | Trace dimensions; explain one attention row and pre-LN ordering |
| 20–25 min | Output-only loss, autograd, training, checkpoint and generation | Explain teacher forcing versus the model's own generated prefix |
| 25–32 min | Model comparison and component findings | Show the same input across real saved checkpoints; distinguish loss from note-level quality |
| 32–38 min | Gradio head/context experiment and limitations | Show exact query/prefix, a controlled change, intervention and category support |
| 38–45 min | Lessons, failure cases and questions | Discuss one unexpected result and what additional evidence would be needed |

For 30 minutes, shorten architecture detail and show one comparison plus one intervention. For 60 minutes, expand the arithmetic, checkpoint workflow and audience questions. These timings are proposals until rehearsal.

## Six-minute UI sequence

1. Open Compare models. Select `final_seed7/transformer_2`, greedy decoding and `PATIENT John Smith VISITED Dr Jones ON January 12`. Generate. State that its fixed checkpoint is update4,000 and its seed is7. The expected target is `PATIENT [NAME] VISITED [DOCTOR] ON [DATE]`; read the actual output and EOS before discussing success.
2. Use `PATIENT Alice White PHONE 5551234567` with both `task_tokens/transformer_2` and `final_seed7/transformer_2`. Preserve the primary42failure beside the correct seed7output. Explain development-only candidate selection; the architecture benchmark remains42.
3. Open Compare components. Load task results, the controlled source pair and seed sensitivity. The24diagnostics use familiar entity/layout pools, while complete50-note final partitions test declared novelty. Read the final primary and separate selected-demo tables with their denominators. Extra heads do not guarantee improvement: the bare four-head model fails the DrugA controlled prediction, while the single-head and full42models pass that pair.
4. Open Inspect heads and choose `task_tokens/transformer_2`, which is seed42. Use source `PATIENT John Smith RECEIVED DrugA` and conditional prefix `PATIENT [NAME] RECEIVED ` with its trailing space. The queried next token is the complete task piece `DrugA`, not its final character. Read the attention row and its exact query.
5. Disable one head and show both actual generations. Load saved seed42ablations: baseline14/14identifier replacements and8/8exact notes, compared with4/14and1/8when L0H0is disabled. These are eight validation notes, not the authored acceptance suite. The chart's checkpoint matters; seed7need not learn the same head roles.
6. Explain one finding using question, change, observation, interpretation and limit. Expand the actual token trace if asked. Use the saved HTML/screenshots if live inference is unavailable, and identify them as recorded evidence.

For the assignment's token-by-token demonstration, expand **Watch actual next-token predictions**, choose `final_seed7/transformer_2` and click **Generate tokens one at a time** using the VISITED source above. The API verification received 12 growing frames ending at EOS and matched the console path exactly. The latest browser visual check is pending. In Compare components, **Final test loss and next-token accuracy** loads all 32 recomputed teacher-forced rows; explain why supplying the correct prefix differs from free generation.

The selected demo's final exact counts are 49, 5, 43 and 5 out of 50 for known/known, known/new, new/known and new/new conditions. The five successes in either new-layout condition are no-identifier controls. Do not describe this as general redaction accuracy. Read FINAL_FINDINGS.md for the component results: the no-residual variant performs better on some measured outcomes, and the tested temperatures all give the same eight exact notes. Neither result should be rewritten to match an expected lesson.

## Each presenter must independently explain

- What information enters the model, and what information it must never read during inference.
- A token, an ID, an embedding vector and a position vector.
- Why attention cannot see future output tokens.
- What changes between the selected architectures.
- Which targets contribute to loss and how a gradient changes weights.
- Why training a different architecture requires its own checkpoint.
- One actual metric/output and a bounded interpretation.

These checks remain pending until each person demonstrates them. Rehearsal is not complete because this script exists.
