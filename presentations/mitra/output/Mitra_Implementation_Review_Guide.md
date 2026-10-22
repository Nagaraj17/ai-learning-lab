# Tiny Mitra implementation review

Evidence snapshot: October 18, 2026. Presentation target: October 23. Proposed main talk: 45 minutes including demonstration and questions, with shorter/longer routes in the project demo script.

This review contains actual local CPU evidence from the fixed 1,000-update, seed-42 comparison. Main generated metrics use 12 validation diagnostics, not the complete final test. Repeats, final partition evaluation and a separately labeled extended-training study are running. Cloud verification and individual presenter explanations remain pending. Do not present the snapshot as final production quality.

Read the consolidated prerequisite guide first in small sessions. The slides support a presentation; the Python notebook provides the detailed calculation walkthrough. For each finding state question, change, observation, interpretation and limit.


## Slide-by-slide explanation and speaking notes

### 1. Tiny Mitra

Building, testing and explaining
a small GPT from PyTorch primitives

Implementation review dated October 18. The main fixed-budget comparison uses eight architectures, seed 42, 1,000 optimizer updates each. This deck reports the 12-note validation generation diagnostics, not complete final-test outcomes or repeated-seed conclusions. Repeats, complete final partitions and extended training are running separately. Individual mastery and actual Colab verification remain pending.

Sources: [task](<C:/Users/Nagar/source/repos/ai-learning-lab/weekly curriculum/06-Tiny-Mitra-Project/TASKS.md>), [protocol](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/protocol.json>).

### 2. Our learning journey

Each week removed a limitation in the previous model

| Stage | What we built or studied | Why it matters for Mitra |
| --- | --- | --- |
| Week 1 | Forward pass, loss and learning | Turn prediction errors into parameter updates |
| Week 2 | Tokenization and embeddings | Represent text as trainable vectors |
| Week 3 | Causal self-attention | Use relevant earlier context |
| Week 4 | Multiple attention heads | Allow several context distributions |
| Week 5 | Transformer blocks and ablations | Test what added components actually buy |

Use this slide as a narrative, not as a list of technologies. A forward pass computes scores, but loss and backpropagation make those scores learnable. Embeddings replace sparse token identities with learned vectors. Attention lets each position gather information from earlier positions. Multiple heads provide separate learned projections. A complete block combines attention with feature transformation, normalization and residual paths. The tracker marks Weeks 1 to 4 complete and Week 5 in progress. That is a record of learning progress, not a fresh assessment of personal mastery. The Week 5 repository contains executable NumPy components and saved experiment results. The Week 6 engineering implementation is now written and checked, but learner walkthroughs remain open. The complete project imports the same Python model from its notebooks and Gradio.

Sources: [progress](<C:/Users/Nagar/source/repos/ai-learning-lab/PROGRESS.md>), [numpy](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/numpy_transformer_suite.py>), [results](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/visualizations/arch_results.json>).

### 3. The synthetic transformation task

Synthetic examples with an explicit redaction policy

```text
PATIENT John Lee TREATMENT DrugA

PATIENT [NAME] TREATMENT DrugA
```

Separate the source, the expected transformation and an actual generated continuation. The goal is to replace the patient identity while preserving the treatment association. Our inference function reads only source plus checkpoint. It cannot call the gold renderer. The task is synthetic and educational; a classifier plus replacement or a rule system is an alternative implementation, and this work does not establish a production advantage.

Sources: [assignment](<C:/Users/Nagar/source/repos/ai-learning-lab/topics/Mitra Topics/Week 6 — Build Your Own Tiny-GPT_Excercise.md>), [data](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/data.py>), [generation](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/generate.py>).

### 4. What the implementation establishes

Engineering evidence and quality evidence answer different questions

| Area | Observed evidence | Remaining boundary |
| --- | --- | --- |
| Mechanics | 20 checks pass; all eight variants are causal | Not a guarantee of task quality |
| Learning loop | Tiny pilot produces 6/6 expected notes | Only four training notes |
| Architecture runs | Eight seed-42 checkpoints; 1,000 updates each | Repeats and final partitions separate |
| Gradio | A/B/C, attention and head APIs tested | Visual rehearsal and team explanation |
| Cloud | Guided notebook and project-only bundle | Actual GPU/resume/CPU test pending |

The tests establish explicit mechanisms and reproducibility properties. The tiny pilot establishes a small source-dependent learning loop, not broad transfer. Current benchmark generations are often poor: report raw failures and clinical preservation. The final claim must be limited to the experiment actually run. The full API check tests outputs, attention rows, intervention/restoration and eight saved rows. Actual cloud runtime use has not occurred.

Sources: [tests](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/mechanics-tests.xml>), [mitra](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/benchmark_summary.json>), [pilot](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/pilot_controlled_context.json>), [ui](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/app.py>), [task](<C:/Users/Nagar/source/repos/ai-learning-lab/weekly curriculum/06-Tiny-Mitra-Project/TASKS.md>).

### 5. A transformation as next-token prediction

The same sequence contains the source and the desired continuation

The displayed text is serialized and then encoded. The character tokenizer retains spaces, case and punctuation and reserves atomic protocol and placeholder tokens. It does not represent PATIENT as one word token. IDs are shifted so the position holding OUTPUT predicts the first output character P. Output tokens and EOS are scored; source prediction and padding targets are ignored. This describes desired targets, not an observed generation.

Sources: [tokenizer](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/tokenizer.py>), [data](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/data.py>), [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>).

### 6. Training and generation

Parallel scoring during training, one appended token during inference

| Aspect | Training | Generation |
| --- | --- | --- |
| Available text | Source and correct output prefix | Source and model-generated prefix |
| Model input | Shifted complete example | Current prompt plus generated tokens |
| Prediction used | All scored output positions | Final position only |
| Update | Loss, backpropagation, optimizer step | Append one token without weight updates |
| Main risk | Target leakage through an incorrect mask | An early error becomes future context |

Training uses expected previous output characters and the source as a causal prefix. Generation begins with only source through OUTPUT, appends one actual selected token, and repeats. Loss is computed only on output targets plus EOS. A correct teacher-forced next token and a correct freely generated note are separate evaluations. An early generated error can feed into later decisions. Checkpoints save weights, tokenizer/configuration, provenance, optimizer, sample-order/random state and completed updates.

Sources: [data](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/data.py>), [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [generation](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/generate.py>).

### 7. Tokenization defines what the model can see



Reuse the Week 3 library analogy carefully: a shelf address is an ID and the stored numerical row is an embedding. The implementation fixes the supported alphabet before final-test use. It does not fit a vocabulary on test examples. Round-trip and offset tests include unseen spellings. Character tokenization makes literal unseen identifiers representable, so zero exact-surface leakage cannot be credited to a word-vocabulary inability to emit them. Nevertheless the leakage scanner can miss partial or reformatted identifiers.

Sources: [tokenizer](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/tokenizer.py>), [tests](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/mechanics-tests.xml>), [guide](<C:/Users/Nagar/source/repos/ai-learning-lab/weekly curriculum/06-Tiny-Mitra-Project/PREREQUISITE_KNOWLEDGE.md>).

### 8. Embeddings and position

A token ID selects a vector, then position adds order information

```text
X[b, t, :] = E[token_id] + P[t]
```

Token embedding E has shape V by 64; learned position table P has shape 512 by 64. At position t, X_t=E[id_t]+P[t]. Corresponding features are added, keeping width 64. Teaching numbers: [0.2,-0.1]+[0.01,0.04]=[0.21,-0.06]. They are invented for arithmetic, not coordinates from a learned clinical feature. Base lookup plus position does not yet incorporate the entire source context; attention does the mixing. Shapes B,T,64 are traced in the shared notebook.

Sources: [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [tokenizer](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/tokenizer.py>), [guide](<C:/Users/Nagar/source/repos/ai-learning-lab/weekly curriculum/06-Tiny-Mitra-Project/PREREQUISITE_KNOWLEDGE.md>).

### 9. Attention retrieves a weighted mixture

Learned compatibility determines how much each earlier value contributes

```text
A = softmax((QKᵀ / √dₕ) + M)    C = AV
```

Q, K and V are different projections of the current hidden representations. A query at one position has a dot product with every eligible key. Transposing K aligns its feature dimension for those comparisons. Dividing by the square root of head width limits score scale under the standard variance intuition. Softmax exponentiates each eligible score and divides by the row sum, yielding nonnegative weights that sum to one. Multiplying by V gives a weighted mixture, not a discrete database lookup. In a causal layer, later keys are ineligible. The query at a generation step belongs to the final prefix position, which predicts the next token. Calling a head a name-detector requires measured evidence. C denotes the context mixture, distinct from the head-count symbol used in shape discussions. In this implementation Q/K/V and output Linear layers include biases. Score masks are explicit; no fused attention or ready-made MultiheadAttention is used.

Sources: [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [attention](<https://arxiv.org/abs/1706.03762>), [tests](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/mechanics-tests.xml>).

### 10. Three masks, three jobs

Illustrative token positions; the model applies the same rule to character tokens

| Query / key | <OUTPUT> | PATIENT | [NAME] | DIAGNOSIS |
| --- | --- | --- | --- | --- |
| <OUTPUT> | allow | block | block | block |
| PATIENT | allow | allow | block | block |
| [NAME] | allow | allow | allow | block |
| DIAGNOSIS | allow | allow | allow | allow |

The colored teaching matrix illustrates eligible query/key relationships. The causal mask hides future keys before softmax. Key padding independently prevents padded tokens from contributing values. The loss mask ignores source-prediction and padded targets, not source representations themselves. Source embeddings can receive gradients through output attention. A future-token perturbation test verifies unchanged earlier logits for every variant.

Sources: [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [data](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/data.py>), [tests](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/mechanics-tests.xml>).

### 11. Multiple heads examine the full sequence

The feature dimension divides across heads, while every head keeps all token positions

| Operation | Illustrative shape | Meaning |
| --- | --- | --- |
| Input | (B, T, 64) | 64 features at every position |
| Q, K, V after split | (B, 4, T, 16) | Four head-specific feature views |
| Per-head attention | (B, 4, T, T) | A separate distribution per query and head |
| Concatenate contexts | (B, T, 64) | Rejoin 4 × 16 features |
| Output projection | (B, T, 64) | Mix information across head outputs |

All eligible earlier positions are accessible to each head. With D=64 and four heads, Dh=16. Q/K/V start B,T,64, reshape B,T,4,16, then transpose B,4,T,16. Scores have B,4,T,T; mixed values B,4,T,16; head features merge back to B,T,64 and pass through an output projection. Head count alone leaves projection parameter count unchanged at fixed width. Learned roles require intervention evidence, not naming heads from a heatmap.

Sources: [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [tests](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/mechanics-tests.xml>).

### 12. A complete decoder block

Our full model normalizes before each sublayer

```text
U = X + Attention(LN₁(X))
Y = U + FFN(LN₂(U))
```

Full blocks use pre-LN: U=X+Attention(LN1(X)), Y=U+FFN(LN2(U)). Our LayerNorm explicitly computes feature mean and population variance with learned scale/offset. FFN expands 64 to 192, applies GELU, then contracts to 64. Two blocks plus final normalization form the full model. Component variants remove the declared branch or additions; the no-FFN branch also removes its otherwise unused pre-normalization parameters. PyTorch supplies primitive arithmetic, Linear/Embedding, autograd and AdamW, not the Transformer implementation.

Sources: [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [tests](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/mechanics-tests.xml>).

### 13. Learning the output continuation

The source contributes context while selected targets contribute loss

```text
L = −Σₜ mₜ log p(yₜ | prefixₜ) / Σₜ mₜ
```

Cross-entropy penalizes low probability for the correct next token. For probabilities [0.75,0.25], a correct first token has loss -ln(0.75)=0.2877; a correct second token has loss 1.3863. Sum losses over scored output/EOS tokens and divide by their total count. Do not average unequal batch means. Autograd computes gradients through the manually assembled network; AdamW updates parameters. Exact CPU interruption/resume matches uninterrupted updates in the test. Learning rate, sampler, optimizer and random state are recorded.

Sources: [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [data](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/data.py>), [tests](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/mechanics-tests.xml>), [protocol](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/protocol.json>).

### 14. Eight trained architecture variants

One shared tokenizer, dataset, width and 1,000-update budget

| Model | Assignment role | Architecture |
| --- | --- | --- |
| Embedding only | A | Token + position embeddings → logits |
| Single attention | B | One bare causal head → logits |
| Multi attention | Additional | Same bare arrangement, four heads |
| One full block | Depth | Pre-LN attention + FFN + residuals |
| Two full blocks | C | Two decoder blocks + final norm |
| No FFN | Ablation | Two blocks without FFN branch |
| No LayerNorm | Ablation | All block/final norms omitted |
| No residual | Ablation | Residual additions omitted |

Each row has its own trained state. Gradio loads the corresponding checkpoint instead of retraining when the selection changes. The bare single/multi variants share the same residual/norm policy. Fixed width keeps the attention projection counts equal across head counts. Component removal changes parameter counts; one versus two blocks also changes capacity. This is a fixed-budget comparison with one primary seed; repeats of four models are separate.

Sources: [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [mitra](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/benchmark_summary.json>), [protocol](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/protocol.json>).

### 15. Measure the transformation

Token accuracy alone can miss leakage and clinical corruption

| Measure | Question | What must accompany it |
| --- | --- | --- |
| Token loss / accuracy | How well is the next token predicted? | Teacher-forced scored-token count |
| Redaction precision / recall | Correct placeholder at the correct field? | TP / FP / FN and category support |
| Exact note match | Was the entire note reproduced correctly? | Number of complete generated notes |
| Clinical preservation | Did the original field/value relation survive? | Correct fields / supported fields |
| Leakage / EOS / latency | What failed or cost time? | Raw text, stop reason and hardware |

The metrics use the declared synthetic field grammar. Placeholder totals alone cannot locate correct redaction, and a clinical keyword appearing somewhere does not establish its original relation. Exact-surface leakage is bounded: it can miss partial or reformatted identifiers. N/A is reported for zero support. Latency from the training work queue is provisional because concurrent jobs can affect it; quiet inference measurement is separate.

Sources: [metrics](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/evaluate.py>), [mitra](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/benchmark_summary.json>), [tests](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/mechanics-tests.xml>).

### 16. Mitra: fixed-budget baseline results

Seed 42; 100 validation notes for loss; 12 generation diagnostics

| Model | Parameters | Val. loss | Redaction F1 | Exact notes |
| --- | --- | --- | --- | --- |
| A: embedding only | 46,829 | 1.618 | 0.000 | 0/12 |
| B: one bare head | 63,469 | 1.800 | 0.000 | 0/12 |
| Four bare heads | 63,469 | 1.570 | 0.160 | 0/12 |
| C: two full blocks | 130,413 | 1.416 | 0.579 | 1/12 |

These are actual Mitra values, not Week 5 numbers. All runs completed 1,000 updates; each model is evaluated at its minimum-validation-loss checkpoint, which can be earlier. The diagnostic generation subset was selected for category support and is small; it is not the complete final test. The full model has better validation loss than the bare baselines but still produces only one exact note out of twelve here. Zero observed exact-surface leakage does not make this successful de-identification.

Sources: [mitra](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/benchmark_summary.json>), [protocol](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/protocol.json>).

### 17. Mitra: component trade-offs

Separate trained variants; one initialization seed

| Variant | Val. loss | Recall | Exact notes | Clinical fields |
| --- | --- | --- | --- | --- |
| transformer_2 | 1.416 | 50.0% | 1/12 | 1/25 |
| no_ffn | 1.578 | 36.4% | 2/12 | 2/25 |
| no_layernorm | 1.431 | 40.9% | 1/12 | 1/25 |
| no_residual | 1.586 | 72.7% | 3/12 | 7/25 |

The full model has the lowest validation loss among these rows, while no-residual generates three exact diagnostics compared with one for the full model. That contradicts a blanket claim that residuals always improve our measured output quality. The no-LayerNorm run has larger logged gradient norms and a late rise in validation loss, but it did not fail numerically. No-FFN changes branch and associated normalization parameter count. Different metrics give different rankings; one seed and twelve notes are insufficient for a universal architecture conclusion.

Sources: [mitra](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/benchmark_summary.json>), [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [protocol](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/protocol.json>).

### 18. Checkpoint selection changes the story

Same full architecture and 12 validation diagnostics

| Selection | Update | Exact notes | Correct redactions | Clinical fields |
| --- | --- | --- | --- | --- |
| Minimum validation loss | 400 | 1/12 | 11/22 | 1/25 |
| Latest training state | 1,000 | 3/12 | 22/22 | 16/25 |

The latest checkpoint was inspected after observing weak validation-only generation. This is explicitly exploratory and does not replace the registered best-loss benchmark. Minimum average cross-entropy over mixed familiar/novel validation conditions need not optimize field-level generated-note quality. The later state improves these particular outputs while average validation loss is worse. The current UI exposes it as transformer_2_latest with its selection/update shown. A separately registered longer-training extension is not included in this table and has not been evaluated on these final-test outputs.

Sources: [mitra](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/benchmark_summary.json>), [latest](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/latest_checkpoint_diagnostic.json>), [protocol](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/protocol.json>).

### 19. Individual heads affect this checkpoint

Full model at update 400; eight validation diagnostics

| Disabled head | Recall after ablation | Change in recall | Exact notes |
| --- | --- | --- | --- |
| L0 H0 | 26.7% | -20.0 pp | 0/8 |
| L0 H1 | 0.0% | -46.7 pp | 0/8 |
| L0 H2 | 46.7% | +0.0 pp | 0/8 |
| L0 H3 | 40.0% | -6.7 pp | 0/8 |
| L1 H0 | 46.7% | +0.0 pp | 1/8 |
| L1 H1 | 33.3% | -13.3 pp | 0/8 |
| L1 H2 | 33.3% | -13.3 pp | 0/8 |
| L1 H3 | 46.7% | +0.0 pp | 0/8 |

Baseline recall is 7/15 and exact match is 1/8. Disabling L0 H1 reduces recall to 0/15. L1 H0 leaves the measured outputs unchanged; that does not establish universal redundancy. Most category supports are tiny, and baseline MRN recall is already zero, so these tests cannot identify a successful MRN head. The selected pair in each block is exploratory, chosen from validation single-head effects. Gates zero the context before output projection; all-one gates reproduce baseline and request-local restoration is tested. Raw per-category precision/recall and teacher-forced loss are saved separately.

Sources: [heads](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/investigations/enriched_head_measurements.json>), [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [tests](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/mechanics-tests.xml>).

### 20. Controlled context: inspect the actual result

One source character changes; output prefix and query position stay fixed

| Checkpoint | Source DrugA | Source DrugB | Scope |
| --- | --- | --- | --- |
| embedding_only | B | B | Expected A / B |
| single_attention | N | N | Expected A / B |
| multi_attention | A | A | Expected A / B |
| transformer_2 | B | B | Expected A / B |
| Tiny four-note pilot | A | B | Separate small grammar |

The main source pair uses PATIENT John Lee DIAGNOSIS NSCLC BIOMARKER EGFR TREATMENT DrugA versus DrugB, with matched output prefix ending TREATMENT Drug. Source token 64 changes and the query index stays fixed. The embedding-only probabilities are exactly invariant, as expected from its mechanism. The primary attention models change probabilities slightly but do not choose the correct A/B pair; H1 is not established by this main probe. The small separately trained pilot uses PATIENT Jo TREATMENT DrugA/B. Its actual observations are shown; do not transfer its success to the main benchmark. A supplied correct prefix is a conditional test, not a claim that the model generated that prefix freely.

Sources: [context](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/investigations/conditional_probabilities.json>), [pilot](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/pilot_controlled_context.json>), [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>).

### 21. Sampling temperature affects reliability

Same update-400 model, eight notes and sampling seeds

| Temperature | Correct redactions | Exact notes | Valid EOS |
| --- | --- | --- | --- |
| 0.1 | 7/15 | 1/8 | 8/8 |
| 0.7 | 7/15 | 0/8 | 8/8 |
| 1.2 | 0/15 | 0/8 | 6/8 |

All three rows use sampling, not a greedy/sampling switch. Each note uses the same base sampling seed under the declared procedure. Higher temperature spreads probability; in these eight trials 1.2 increases malformed or incomplete behavior, but one sampling seed and eight prompts do not estimate a general distribution. Low positive temperature is still random. Greedy decoding is a separate reference and is invariant to positive logit scaling. OOD stress prompts have no gold labels and are qualitative.

Sources: [generation](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/generate.py>), [heads](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/investigations/enriched_head_measurements.json>).

### 22. Week 5: a modest measured improvement

Saved next-token evaluation on synthetic workflow sequences

| Architecture | Parameters | Accuracy: mean ± SD | Test loss | Macro F1 |
| --- | --- | --- | --- | --- |
| A: embedding + linear | 1,911 | 69.69% ± 0.77 | 1.547 | 0.407 |
| B: one attention head | 4,239 | 64.06% ± 3.56 | 2.073 | 0.355 |
| C: four attention heads | 4,239 | 62.81% ± 3.62 | 2.234 | 0.326 |
| D: two full blocks | 16,215 | 70.94% ± 1.25 | 1.478 | 0.442 |

Read these as small-sample evidence from the saved Week 5 experiment, not Mitra performance. Across five initialization seeds, the full two-block model D improves mean accuracy by 1.25 percentage points over A. The test set in each saved run contains 64 target tokens across four cases. The evaluated scenario labels include both held-out families and familiar families. The saved JSON does not include a full dataset manifest or histories, so it does not establish a broad reproducibility claim. Mean test loss falls from about 1.547 to 1.478 and macro F1 rises from about 0.407 to 0.442. The standard deviations are across seeds, not confidence intervals. No statistical significance claim follows from these aggregates. This is a different task and vocabulary. Its token accuracy is not directly comparable to character-token Mitra accuracy. Use it to show the experimental method, not a before/after product metric.

Sources: [results](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/visualizations/arch_results.json>), [runner](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/experiment_runner.py>).

### 23. Demonstrate three kinds of evidence

Live continuations and saved measurements are labeled separately

| Gradio view | Action | Explanation |
| --- | --- | --- |
| Compare models | Same synthetic note; select A/B/C or latest | Show actual continuation, checkpoint and stop |
| Inspect heads | One exact query; then disable one head | Routing pattern versus intervention effect |
| Compare components | Load saved curves, reports and partitions | Separate retrained variants and sample sizes |

Use a successful validation example and a failure example, clearly identified as selected examples. Never replace a generated string with its expected label. The latest full checkpoint is exploratory and labeled separately from the best-loss comparison. Head charts identify their saved checkpoint so they cannot be silently attributed to another live selection. The final-partition table loads only completed 50-note rows. Personal presenter names, exact duration and individual walkthroughs remain pending.

Sources: [ui](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/app.py>), [task](<C:/Users/Nagar/source/repos/ai-learning-lab/weekly curriculum/06-Tiny-Mitra-Project/TASKS.md>), [mitra](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/benchmark_summary.json>).

### 24. What this work teaches us

A GPT learns parameters that turn a prefix into the next-token distribution

Do not conclude that this is a production de-identification system or that every team member has demonstrated mastery. The engineering contribution is a working manually assembled model, reproducible comparisons and bounded investigations. The learning contribution must be completed through each person explaining a prompt and interpreting actual evidence. The next exercises on scaffolding/fine-tuning remain separate.

Sources: [task](<C:/Users/Nagar/source/repos/ai-learning-lab/weekly curriculum/06-Tiny-Mitra-Project/TASKS.md>), [mitra](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/benchmark_summary.json>), [latest](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/latest_checkpoint_diagnostic.json>).

### 25. Appendix: attention with small numbers

Illustrative arithmetic, not an observed attention head

| Step | Calculation | Result |
| --- | --- | --- |
| Query and keys | q = [1, 0], k₁ = [1, 0], k₂ = [0, 1] | Head width dₕ = 2 |
| Scaled scores | [q·k₁, q·k₂] / √2 | [0.7071, 0.0000] |
| Softmax weights | [exp(0.7071), exp(0)] / sum | [0.6698, 0.3302] |
| Value vectors | v₁ = [2, 0], v₂ = [0, 4] | Two vectors of width 2 |
| Weighted context | a₁ × v₁ + a₂ × v₂, using unrounded weights | [1.3395, 1.3210] |

A dot product multiplies matching components and adds them. With q=[1,0], the first key scores 1 and the second scores 0. Dividing by square root 2 gives about 0.7071 and 0. Exponentiation gives positive values, and division by their sum yields probabilities 0.66976155 and 0.33023845. Multiplying those weights by the two value vectors produces [1.33952310,1.32095380]. The output is a mixture and need not equal any original value vector. These are invented inputs used for a verified arithmetic explanation. They are not evidence that a trained Mitra head attends to a name.

Sources: [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [attention](<https://arxiv.org/abs/1706.03762>).

### 26. Appendix: normalization and the FFN

Operations within a token representation

```text
LN(x) = γ ⊙ (x − μ) / √(σ² + ε) + β
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

LayerNorm uses features within one position, not statistics across all examples in a batch. With x=[1,3], mean is 2 and population variance is 1. Ignoring epsilon for this arithmetic and using gamma=[1,1], beta=[0,0], normalized output is [-1,1]. Learned gamma and beta can rescale and shift it. For the FFN, W1 has shape (64,192), b1 has shape (192), W2 has shape (192,64), and b2 has shape (64). The nonlinearity is essential: two linear maps with no nonlinear activation collapse into a single linear map. Our implementation uses GELU while the earlier NumPy study uses ReLU. Avoid claiming expansion itself guarantees richer measured behavior.

Sources: [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [numpy](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/numpy_transformer_suite.py>), [layernorm](<https://arxiv.org/abs/1607.06450>).

### 27. Appendix: the loss-mask boundary

Excerpt under the actual character tokenizer

| Input token | Next target | Score? | Reason |
| --- | --- | --- | --- |
| <INPUT> | P | No | Source prediction |
| Last source character | <OUTPUT> | No | Protocol boundary |
| <OUTPUT> | P | Yes | First output character |
| P | A | Yes | Next output character |
| Last output character | <EOS> | Yes | Learn termination |
| Padding | -100 (ignore) | No | Batch filler |

This is a readable excerpt, not one contiguous six-token example. Inputs omit the final serialized token; targets omit the first. At OUTPUT, score the first output character P. Ignore source-prediction targets and padding but retain source attention dependencies. The target ignore index is -100, not a token ID. EOS is always scored. Context overflow is rejected so the output/EOS cannot disappear silently.

Sources: [data](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/data.py>), [tokenizer](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/tokenizer.py>), [tests](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/mechanics-tests.xml>).

### 28. Appendix: measured parameter count

Full model: V=109, width=64, FFN=192, context=512

| Component | Calculation | Parameters |
| --- | --- | --- |
| Token embedding | 109 × 64 | 6,976 |
| Position embedding | 512 × 64 | 32,768 |
| Attention, two blocks | 2 × 4 × (64² + 64) | 33,280 |
| FFNs, two blocks | 2 × (2 × 64 × 192 + 192 + 64) | 49,664 |
| All norms | 5 × (64 + 64) | 640 |
| Vocabulary head | 64 × 109 + 109 | 7,085 |
| Total | Sum of the six components | 130,413 |

Actual parameter_count is 130,413. All attention projections and the untied vocabulary head include biases. The position table is 512 positions even though the saved dataset maximum full length is about 140 tokens. Unlike the former starter example, this is the measured project configuration with a fixed character vocabulary of 109 entries. Four heads split the same total width and do not multiply projection parameter count.

Sources: [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [config](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/config.py>), [mitra](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/benchmark_summary.json>).

### 29. Appendix: the autoregressive loop

Conceptual pseudocode, not a recorded run

```text
ids = encode(source_prompt_through_OUTPUT)
while generated_count < limit:
    logits = model(ids)
    next_logits = logits[:, -1, :]
    next_id = argmax(next_logits)       # explicit greedy mode
    ids = append(ids, next_id)
    if next_id == EOS:
        break
```

Explain every line in order. Encode the entire source prompt including <OUTPUT>. The model returns vocabulary logits for all supplied positions. Slice the last position and select a single next token. Append it to the supplied context before the next forward pass. EOS is a learned token, and a maximum output length prevents an unbounded loop. The conceptual example deliberately separates greedy selection from temperature-based sampling. Without KV caching, each iteration recomputes prior representations. Attention score work per layer is quadratic in current context length, and total recomputation across many new tokens grows faster than the cost of one step. Detect context overflow explicitly rather than silently discarding the source.

Sources: [generation](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/generate.py>), [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>).

### 30. Appendix: generalization conditions

Final partitions are separate from validation-only diagnostics

| Partition | Notes | Condition |
| --- | --- | --- |
| Known names / known layouts | 50 | New combinations; familiar pools |
| New names / known layouts | 50 | Separate final-name pool |
| Known names / new layouts | 50 | Final-only structural templates |
| New names / new layouts | 50 | Joint entity/layout transfer |
| Training / validation | 700 / 100 | Validation includes its own novel conditions |

Final-test evaluation is released only after a saved review of the fixed protocol. It evaluates every note in each 50-example partition for each of the eight primary seed-42 best-loss models. Complete final outcomes and repeated-seed summaries are separate running work at this review. No-identifier controls share a layout, dates can recur and novelty for doctors/addresses differs; do not call all identifiers wholly unseen. The tokenizer supports new ASCII spellings without fitting on final answers. No final-test outcomes may tune the reported runs.

Sources: [data](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/data.py>), [protocol](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/protocol.json>), [task](<C:/Users/Nagar/source/repos/ai-learning-lab/weekly curriculum/06-Tiny-Mitra-Project/TASKS.md>).

### 31. Appendix: questions for the technical discussion



Explain the distinction between an optimization criterion and the task-level generated metrics. The later full checkpoint improves these twelve validation outputs but is not silently substituted into the fixed-budget benchmark. A supplied prefix is not free generation. A head ablation measures dependence of one checkpoint on one intervention under the tested sample; it does not assign an invariant semantic role. Claims about novel templates need the declared complete partitions, not the familiar diagnostic subset. Each presenter must explain these points independently before the presentation is marked rehearsed.

Sources: [latest](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/latest_checkpoint_diagnostic.json>), [heads](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/investigations/enriched_head_measurements.json>), [protocol](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/protocol.json>), [task](<C:/Users/Nagar/source/repos/ai-learning-lab/weekly curriculum/06-Tiny-Mitra-Project/TASKS.md>).

### 32. Appendix: evidence and references



Primary sources are recorded in the relevant slide notes. Main project evidence: C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/benchmark_summary.json . Manual implementation: C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py . Registered protocol: C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/protocol.json . Week 5 evidence: C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/visualizations/arch_results.json . Architecture: https://arxiv.org/abs/1706.03762 . Normalization: https://arxiv.org/abs/1607.06450 . Attention interpretation: https://aclanthology.org/N19-1357/ and https://aclanthology.org/D19-1002/ . No claims of actual cloud execution, clinical readiness, final repeated-seed conclusions or completed individual mastery are made.

Sources: [mitra](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/benchmark_summary.json>), [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [protocol](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/protocol.json>), [tests](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/mechanics-tests.xml>), [results](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/visualizations/arch_results.json>), [attention](<https://arxiv.org/abs/1706.03762>), [layernorm](<https://arxiv.org/abs/1607.06450>), [interpretation](<https://aclanthology.org/N19-1357/>), [interpretation2](<https://aclanthology.org/D19-1002/>).
