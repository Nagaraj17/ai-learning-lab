# Tiny Mitra presenter guide

Prepared for the October 23 presentation. The core talk has 28 slides plus 8 appendix slides. A 45-minute rehearsal is provisional until team roles and the exact slot are confirmed. The primary architecture benchmark uses seed 42; the selected demo uses its explicitly labelled development candidate. Every presenter must independently trace a prompt. Actual Colab verification is still pending.

Use START_HERE.md for a short first session and PREREQUISITE_KNOWLEDGE.md for the complete refresher. Read the source/expected/generated distinction before rehearsing. Never repair a model output for the audience. The deck reports actual final corpus evidence; earlier experiments remain separately recorded.

## Slide-by-slide explanation and speaking notes

### 1. Tiny Mitra

A GPT built from primitives
for synthetic note transformation

Presentation scheduled for 23 October 2026. Explain our conditional generation task, then trace the full token path and examine actual experiments. All primary architecture comparisons use seed 42 and the fixed 4,000th update on one saved corpus. The selected live demo is separately labelled final_seed7/transformer_2. It passed 11 authored development notes and 2 clinical checks. The two extra clinical sources occur in training. Held-out outcomes are measured separately. Actual cloud execution and individual presenter teach-back remain human gates.

Sources: [protocol](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/protocol.json>), [demo](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/reports/DEMO_SELECTION.json>), [demoFinal](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/demo_final_test/summary.json>).

### 2. Our learning journey

Each week removed a limitation in the previous model

| Stage | What we built or studied | Why it matters for Mitra |
| --- | --- | --- |
| Week 1 | Forward pass, loss and learning | Turn prediction errors into parameter updates |
| Week 2 | Tokenization and embeddings | Represent text as trainable vectors |
| Week 3 | Causal self-attention | Use relevant earlier context |
| Week 4 | Multiple attention heads | Allow several context distributions |
| Week 5 | Transformer blocks and ablations | Test what added components actually buy |

Our earlier assignments connect directly to Mitra. Week 1 established forward/loss/backward/update. Tokenization and embeddings gave numerical input representations. Attention allowed positions to use earlier context; multiple heads supplied different learned projections. Week 5 compared complete blocks and ablations. Mitra adds a shared PyTorch implementation, real training/checkpoints, output-only supervision and a UI. These are engineering connections. Each presenter still needs to demonstrate understanding independently.

Sources: [progress](<C:/Users/Nagar/source/repos/ai-learning-lab/PROGRESS.md>), [numpy](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/numpy_transformer_suite.py>), [results](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/visualizations/arch_results.json>).

### 3. The synthetic transformation task

Synthetic examples with an explicit redaction policy

```text
PATIENT John Smith VISITED Dr Jones ON January 12

PATIENT [NAME] VISITED [DOCTOR] ON [DATE]
```

This slide shows an expected target under our synthetic assignment policy. Patient name, physician including its title, and date become typed placeholders. Wording such as VISITED and ON survives. Other examples retain diagnosis, biomarker and treatment associations. Generation reads the source and learned weights; it does not call the annotated renderer or look up this answer. A compact grammar also admits deterministic alternatives. Our reason for building GPT is to understand the entire next-token mechanism.

Sources: [assignment](<C:/Users/Nagar/source/repos/ai-learning-lab/topics/Mitra Topics/Week 6 — Build Your Own Tiny-GPT_Excercise.md>), [broad](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/broad_data.py>), [generation](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/generate.py>).

### 4. Final experiment design

One corpus and the same declared checkpoint rule

| Setting | Primary comparison | Scope |
| --- | --- | --- |
| Data | 700 training /100 validation /200 final | 1,000 unique synthetic source notes |
| Model | Width 64, FFN 192, context 256, vocabulary 146 | Eight declared architectures |
| Optimization | AdamW, lr 0.0007, batch 16,4,000 updates | Random initial weights, seed 42 |
| Checkpoint | Final update 4,000 for every model | Minimum-loss weights retained separately |
| Repeats | Four architectures, seeds 7 and 123 | Initialization and minibatch order vary |

The dataset contains ten training sentence forms, names, dates, MRNs, phones, addresses and clinical fields. Exact authored development sources and reserved numerical values are excluded from this final corpus. Final test names exclude development names, including Olivia. All models start from random weights on the same corpus; no earlier experiment supplies their weights. Equal update counts and fixed width do not imply equal parameters or equal compute. Earlier character and continued-training studies remain separate histories and are not pooled into this comparison. The primary source/data/producer/protocol identities were frozen before final evaluation. Do not select checkpoints from test outcomes.

Sources: [protocol](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/protocol.json>), [dataset](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/dataset/manifest.json>), [broad](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/broad_data.py>), [repeats](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/repeats_summary.json>).

### 5. A transformation as next-token prediction

The same sequence contains the source and the desired continuation

Serialize the source, boundary marker and desired output. The final tokenizer has 146 entries: the original alphabet/control vocabulary plus 37 declared task pieces. PATIENT is a single task token. John still uses J, o, h, n character tokens. Spaces remain tokens. At the input position holding OUTPUT, the desired next token is PATIENT. Source-prediction targets are ignored, while output tokens and EOS are scored. The correct target prefix is available during training only.

Sources: [tokenizer](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/task_tokenizer.py>), [data](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/data.py>), [runtime](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/task_runtime.py>).

### 6. Training and generation

Parallel scoring during training, one appended token during inference

| Aspect | Training | Generation |
| --- | --- | --- |
| Available text | Source and correct output prefix | Source and model-generated prefix |
| Model input | Shifted complete example | Current prompt plus generated tokens |
| Prediction used | All scored output positions | Final position only |
| Update | Loss, backpropagation, optimizer step | Append one token without weight updates |
| Main risk | Target leakage through an incorrect mask | An early error becomes future context |

Training uses expected previous output tokens and the source as a causal prefix. Generation begins with only source through OUTPUT, appends one actual selected token, and repeats. Loss is computed only on output targets plus EOS. A correct teacher-forced next token and a correct freely generated note are separate evaluations. An early generated error can feed into later decisions. Checkpoints save weights, tokenizer/configuration, provenance, optimizer, sample-order/random state and completed updates. The same model.py supplies both paths. No architecture or optimizer update occurs when the audience clicks Generate in Gradio.

Sources: [data](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/data.py>), [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [generation](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/generate.py>).

### 7. Tokens, IDs and unseen spelling



An integer ID locates a row in a learned embedding table; its numeric magnitude is not semantic distance. The fixed tokenizer makes PATIENT one token and retains unsupported-as-a-piece spelling as ASCII characters. May can match inside Maya, because longest-piece matching is not whole-word tokenization. Encoding and decoding preserve the exact supported text and offsets. Unsupported Unicode raises an explicit error. The vocabulary is declared independently of final data. Training-name coverage checks ensure that all 52 letter IDs actually occur, but representability and exposure do not establish reliable generalization. The character teaching pilot and historical studies use a different token unit.

Sources: [tokenizer](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/task_tokenizer.py>), [dataset](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/dataset/manifest.json>), [tests](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/mechanics-tests.xml>), [guide](<C:/Users/Nagar/source/repos/ai-learning-lab/weekly curriculum/06-Tiny-Mitra-Project/PREREQUISITE_KNOWLEDGE.md>).

### 8. Embeddings and position

A token ID selects a vector, then position adds order information

```text
X[b, t, :] = E[token_id] + P[t]
```

Token embedding E has shape 146 by 64; learned position table P has shape 256 by 64. At position t, X_t=E[id_t]+P[t]. Corresponding features are added, keeping width 64. Teaching numbers: [0.2,-0.1]+[0.01,0.04]=[0.21,-0.06]. They are invented for arithmetic, not coordinates from a learned clinical feature. Base lookup plus position does not yet incorporate the entire source context; attention does the mixing. Shapes B, T,64 are traced in the shared notebook. The saved corpus maximum serialized sequence has 86 tokens. Allocating 256 position rows does not teach every position or guarantee accuracy on longer notes. The arithmetic shown uses illustrative vectors, not learned clinical meanings.

Sources: [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [tokenizer](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/task_tokenizer.py>), [guide](<C:/Users/Nagar/source/repos/ai-learning-lab/weekly curriculum/06-Tiny-Mitra-Project/PREREQUISITE_KNOWLEDGE.md>).

### 9. Attention retrieves a weighted mixture

Learned compatibility determines how much each earlier value contributes

```text
A = softmax((QKᵀ / √dₕ) + M)    C = AV
```

Q, K and V are different projections of the current hidden representations. A query at one position has a dot product with every eligible key. Transposing K aligns its feature dimension for those comparisons. Dividing by the square root of head width limits score scale under the standard variance intuition. Softmax exponentiates each eligible score and divides by the row sum, yielding nonnegative weights that sum to one. Multiplying by V gives a weighted mixture, not a discrete database lookup. In a causal layer, later keys are ineligible. The query at a generation step belongs to the final prefix position, which predicts the next token. Calling a head a name-detector requires measured evidence. C denotes the context mixture, distinct from the head-count symbol used in shape discussions. In this implementation Q/K/V and output Linear layers include biases. Score masks are explicit; no fused attention or ready-made MultiheadAttention is used.

Sources: [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [attention](<https://arxiv.org/abs/1706.03762>), [tests](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/mechanics-tests.xml>).

### 10. Three masks, three jobs

Visibility masks act on attention, while the loss mask chooses targets

| Query / key | <OUTPUT> | PATIENT | [NAME] | DIAGNOSIS |
| --- | --- | --- | --- | --- |
| <OUTPUT> | allow | block | block | block |
| PATIENT | allow | allow | block | block |
| [NAME] | allow | allow | allow | block |
| DIAGNOSIS | allow | allow | allow | allow |

The colored teaching matrix illustrates eligible query/key relationships. The causal mask hides future keys before softmax. Key padding independently prevents padded tokens from contributing values. The loss mask ignores source-prediction and padded targets, not source representations themselves. Source embeddings can receive gradients through output attention. A future-token perturbation test verifies unchanged earlier logits for every variant.

Sources: [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [data](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/data.py>), [tests](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/mechanics-tests.xml>).

### 11. Heads share positions and split features

The feature dimension divides across heads, while every head keeps all token positions

| Operation | Illustrative shape | Meaning |
| --- | --- | --- |
| Input | (B, T, 64) | 64 features at every position |
| Q, K, V after split | (B, 4, T, 16) | Four head-specific feature views |
| Per-head attention | (B, 4, T, T) | A separate distribution per query and head |
| Concatenate contexts | (B, T, 64) | Rejoin 4 × 16 features |
| Output projection | (B, T, 64) | Mix information across head outputs |

All eligible earlier positions are accessible to each head. With D=64 and four heads, Dh=16. Q/K/V start B, T,64, reshape B, T,4,16, then transpose B,4, T,16. Scores have B,4, T, T; mixed values B,4, T,16; head features merge back to B, T,64 and pass through an output projection. Head count alone leaves projection parameter count unchanged at fixed width. Learned roles require intervention evidence, not naming heads from a heatmap.

Sources: [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [tests](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/mechanics-tests.xml>).

### 12. Two decoder blocks

Our full model normalizes before each sublayer

```text
U = X + Attention(LN₁(X))
Y = U + FFN(LN₂(U))
```

Full blocks use pre-LN: U=X+Attention(LN 1(X)), Y=U+FFN(LN 2(U)). Our LayerNorm explicitly computes feature mean and population variance with learned scale/offset. FFN expands 64 to 192, applies GELU, then contracts to 64. Two blocks plus final normalization form the full model. Component variants remove the declared branch or additions; the no-FFN branch also removes its otherwise unused pre-normalization parameters. PyTorch supplies primitive arithmetic, Linear/Embedding, autograd and AdamW, not the Transformer implementation.

Sources: [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [tests](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/mechanics-tests.xml>).

### 13. Learning the output continuation

The source contributes context while selected targets contribute loss

```text
L = −Σₜ mₜ log p(yₜ | prefixₜ) / Σₜ mₜ
```

Cross-entropy penalizes low probability for the correct next token. For probabilities [0.75,0.25], a correct first token has loss -ln(0.75)=0.2877; a correct second token has loss 1.3863. Sum losses over scored output/EOS tokens and divide by their total count. Do not average unequal batch means. Autograd computes gradients through the manually assembled network; AdamW updates parameters. Exact CPU interruption/resume matches uninterrupted updates in the test. Learning rate, sampler, optimizer and random state are recorded. Gradients reach source embeddings through the attention dependency even though source-prediction targets are ignored. Clipping limits total gradient norm to 1 before the AdamW update. The logged norm is measured before clipping. CPU resume tests compare interrupted and uninterrupted parameter states exactly; CUDA recovery remains unexecuted.

Sources: [runtime](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/task_runtime.py>), [data](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/data.py>), [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [tests](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/mechanics-tests.xml>).

### 14. Eight trained architecture variants

Every choice loads its own trained checkpoint

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

Each row has its own trained state. Gradio loads the corresponding checkpoint instead of retraining when the selection changes. The bare single/multi variants share the same residual/norm policy. Fixed width keeps the attention projection counts equal across head counts. Component removal changes parameter counts; one versus two blocks also changes capacity. This is a fixed-budget comparison with one primary seed; repeats of four models are separate. No-LayerNorm removes all block norms and the final norm. No-FFN removes each FFN and its otherwise unused preceding norm. No-residual removes attention and FFN residual additions but keeps their sublayers. The bare single/four-head models contain no full decoder blocks. These distinctions are implemented explicitly in model.py.

Sources: [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [mitra](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/summary.json>), [protocol](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/protocol.json>).

### 15. Transformation metrics

Token accuracy alone can miss leakage and clinical corruption

| Measure | Question | What must accompany it |
| --- | --- | --- |
| Token loss / accuracy | How well is the next token predicted? | Teacher-forced scored-token count |
| Redaction precision / recall | Correct placeholder at the correct field? | TP / FP / FN and category support |
| Exact note match | Was the entire note reproduced correctly? | Number of complete generated notes |
| Clinical preservation | Did the original field/value relation survive? | Correct fields / supported fields |
| Leakage / EOS / latency | What failed or cost time? | Raw text, stop reason and hardware |

Teacher-forced loss and token accuracy score output/EOS targets, using all 100 validation notes. Free generation compares actual notes with annotations. Redaction recall is TP/(TP+FN); precision is TP/(TP+FP); F 1 is their harmonic mean when defined. TP requires the right placeholder in the right sentence slot. Duplicate/wrong-field placeholders can add false positives. Clinical preservation requires the original field/value association, not a keyword somewhere in output. Exact match judges the entire string. Report category supports and zero-support N/A. Exact-surface leakage can miss partial or reformatted identifiers. Quiet CPU latency is measured separately from training jobs and includes varying generated lengths. No live score is invented for an arbitrary unannotated input.

Sources: [metrics](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/assignment_evaluate.py>), [runtime](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/task_runtime.py>), [latency](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/quiet_cpu_latency.json>).

### 16. Loss and next-token accuracy

Teacher forcing supplies the correct prefix; free generation remains a separate test

| Architecture | Last batch loss | Validation loss | Familiar test: loss / accuracy | New layout: loss / accuracy |
| --- | --- | --- | --- | --- |
| Embedding only | 0.937 | 1.801 | 1.082 / 66.7% | 2.245 / 52.6% |
| One attention head | 0.00746 | 4.717 | 0.163 / 97.0% | 6.878 / 57.2% |
| Four attention heads | 0.00160 | 3.541 | 0.198 / 96.8% | 5.898 / 56.9% |
| Two full blocks | 0.000165 | 1.915 | 0.024 / 99.4% | 2.875 / 64.4% |

Training loss is the last minibatch at update 4000, not a 700-note mean. Validation loss covers all 100 notes. Each test column covers 50 notes; averages weight scored output/EOS tokens, not notes. The familiar test has 532 scored tokens; the new-layout test has its separately recorded support. Teacher-forced accuracy can be high even when free generation fails: the correct prefix is supplied at each prediction. Loss is in natural-log units per task token; do not compare it numerically with historical character/word token losses. Full 32-row recomputation matches the saved final metrics, weights and data.

Sources: [primary](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/summary.json>), [partitions](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/final_test/summary.json>), [runtime](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/task_runtime.py>), [metrics](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/assignment_evaluate.py>), [testTokens](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/final_test/token_metrics_audit.json>).

### 17. Architecture comparison

Primary seed 42, fixed 4,000 updates,24 familiar-condition validation notes

| Architecture | Exact notes /24 | Clinical associations /24 |
| --- | --- | --- |
| Embedding | 2/24 | 1/24 |
| 1 head | 22/24 | 23/24 |
| 4 heads | 20/24 | 21/24 |
| 1 block | 23/24 | 24/24 |
| 2 blocks | 23/24 | 23/24 |
| No FFN | 23/24 | 24/24 |
| No norm | 23/24 | 23/24 |
| No residual | 24/24 | 24/24 |

Embedding only:exact 2/24, identifiers 9/36, clinical 1/24, parameters 35218. One attention head:exact 22/24, identifiers 33/36, clinical 23/24, parameters 51858. Four attention heads:exact 20/24, identifiers 34/36, clinical 21/24, parameters 51858. One full block:exact 23/24, identifiers 35/36, clinical 24/24, parameters 77074. Two full blocks:exact 23/24, identifiers 36/36, clinical 23/24, parameters 118802. Without FFN:exact 23/24, identifiers 35/36, clinical 24/24, parameters 68882. Without LayerNorm:exact 23/24, identifiers 36/36, clinical 23/24, parameters 118162. Without residuals:exact 24/24, identifiers 36/36, clinical 24/24, parameters 118802. All 24 category-selected diagnostic notes use known entity/layout pools. They are validation examples, not the novel-name/layout test partitions. The chart shows counts out of 24 notes and 24 clinical associations; identifiers have 36 supports and remain in the report. Interpret metric rankings separately. The embedding model lacks contextual mixing, but adding heads or blocks does not guarantee a measured improvement under one budget.

Sources: [primary](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/summary.json>), [metrics](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/assignment_evaluate.py>), [protocol](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/protocol.json>).

### 18. Component removal experiments

Separate training runs under the same seed 42 policy

| Variant | Parameters | Exact notes /24 | Clinical /24 | Max logged gradient |
| --- | --- | --- | --- | --- |
| Two full blocks | 118,802 | 23/24 | 23/24 | 1.38 |
| Without LayerNorm | 118,162 | 23/24 | 23/24 | 2.69 |
| Without residuals | 118,802 | 24/24 | 24/24 | 3.11 |
| Without FFN | 68,882 | 23/24 | 24/24 | 1.39 |

The full and no-residual models have the same parameter count, while no-FFN/no-LayerNorm remove different parameter groups. Maximum gradients are sampled at validation logging intervals, before clipping, not a count of every update or a proof that a run diverged. All declared runs completed; inspect curves alongside generated-note quality. The no-LayerNorm variant removes the final norm too. We do not tune a separate learning rate for each ablation. A one-seed component difference is exploratory, and architecture plus optimization can interact.

Sources: [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [comparison](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/descriptive_comparison.json>), [primary](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/summary.json>).

### 19. Training-seed sensitivity

Same development notes, different initialization and minibatch order

| Architecture | Seed 42 exact | Seed 7 exact | Seed 123 exact | Authored range |
| --- | --- | --- | --- | --- |
| Embedding only | 2/24 | 1/24 | 2/24 | 0–0/11 |
| One attention head | 22/24 | 22/24 | 18/24 | 9–11/11 |
| Four attention heads | 20/24 | 24/24 | 22/24 | 8–9/11 |
| Two full blocks | 23/24 | 24/24 | 24/24 | 10–11/11 |

Three training seeds share the same corpus and evaluation notes. They are not three independent clinical datasets. Exact counts are shown rather than p-values. Reports also retain mean/range for recall and clinical preservation. The primary architecture benchmark remains seed 42. final_seed7/transformer_2 is a separately labelled demo selected by the declared development acceptance rule, with priority 42 then 7 then 123. We declared that rule after observing primary 42 at 10/11 and before either repeat completed. It requires 11 authored/EOS plus 2 clinical checks. Both extra clinical sources occur in training. Final-test outcomes do not choose a demo seed.

Sources: [repeats](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/repeats_summary.json>), [comparison](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/descriptive_comparison.json>), [demo](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/reports/DEMO_SELECTION.json>).

### 20. Individual head interventions

Full seed 42 at 4,000 updates, same eight validation notes

| Disabled head | Correct identifiers | Exact notes | Conditional loss |
| --- | --- | --- | --- |
| L0 H0 | 4/14 | 1/8 | 1.1057 |
| L0 H1 | 10/14 | 6/8 | 0.3461 |
| L0 H2 | 12/14 | 5/8 | 0.3369 |
| L0 H3 | 14/14 | 7/8 | 0.0268 |
| L1 H0 | 13/14 | 7/8 | 0.0637 |
| L1 H1 | 14/14 | 7/8 | 0.0193 |
| L1 H2 | 13/14 | 6/8 | 0.1524 |
| L1 H3 | 14/14 | 7/8 | 0.0995 |

Baseline 8/8 exact,14/14 identifiers,6/6 clinical, conditional loss 0.0002. A gate zeros one head context before merge and output projection; weights stay unchanged. L 0 H 0 removal gives 4/14 identifiers and 1/8 notes. Some heads preserve recall while worsening exact notes and loss. Category supports are PATIENT 7, MRN 2, and one each for doctor, date, address, DOB, phone. A heatmap or drop on 2 MRNs does not establish a semantic MRN head. Two pairs 0+1 per block were fixed before intervention; pair effects are non-additive. The runner verifies baseline restoration. These heads belong to seed 42, even when the live demo uses seed 7.

Sources: [heads](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/investigations/enriched_head_measurements.json>), [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [tests](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/mechanics-tests.xml>).

### 21. Controlled source dependence

Change DrugA to DrugB while keeping prefix and position fixed

| Checkpoint | DrugA source | DrugB source | P(DrugA), A / B sources |
| --- | --- | --- | --- |
| Embedding only | [MRN] | [MRN] | 0.002 / 0.002 |
| One attention head | DrugA | DrugB | 0.448 / <0.001 |
| Four attention heads | Chemotherapy | DrugB | 0.089 / <0.001 |
| Two full blocks | DrugA | DrugB | 0.970 / <0.001 |

Each source has the same number of tokens, and the supplied output prefix ends PATIENT [NAME] RECEIVED followed by a space. The next decision is an atomic DrugA/DrugB token. This separates probability-level context use from free generation. Read the actual selected tokens and both probabilities. An embedding-only model sees only the final position token/vector at its vocabulary head, so matching final token and position gives matching probabilities. The full attention model can route earlier source information. A successful controlled pair demonstrates this local dependence; it does not establish novel-template accuracy. The prefix is supplied conditionally, and the report separately retains actual free generations.

Sources: [context](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/investigations/architecture_context.json>), [tokenizer](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/task_tokenizer.py>), [generation](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/generate.py>).

### 22. Sampling temperature

Same seed 42 checkpoint, eight notes and matched sampling seeds

| Temperature | Correct identifiers | Exact notes | Valid EOS |
| --- | --- | --- | --- |
| 0.1 | 14/14 | 8/8 | 8/8 |
| 0.7 | 14/14 | 8/8 | 8/8 |
| 1.2 | 14/14 | 8/8 | 8/8 |

Every row samples from softmax(logits/temperature); this is not a switch between greedy and sampling. A larger positive temperature spreads the probability distribution. Read observed counts rather than assuming a monotonic result. These are eight sampled notes under one seed per note, not a robust distribution estimate. Greedy argmax is a separate mode and ignores the temperature slider. The qualitative out-of-distribution probes have no annotated gold accuracy.

Sources: [generation](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/generate.py>), [heads](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/investigations/enriched_head_measurements.json>).

### 23. Out-of-distribution substitution

Primary seed 42, actual greedy generation

```text
PATIENT Kai DIAGNOSIS UNKNOWN

PATIENT [NAME] DIAGNOSIS DIABETES
```

The source says UNKNOWN; the model substitutes DIABETES. It assigns about 0.811 probability to that selected token at this step. A learned continuation preference is not confidence calibrated to correctness or evidence in the source. The training objective rewards likely next tokens; it does not provide a mechanism to verify unfamiliar facts. These are authored qualitative probes, not a scored OOD benchmark. The actual result is preserved without repair. More varied data, explicit task constraints and broader evaluation would be needed before considering operational use.

Sources: [generation](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/generate.py>), [heads](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/investigations/enriched_head_measurements.json>), [ood](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/investigations/ood.json>).

### 24. Transfer across held-out conditions

Complete 50-note partitions, primary full model and selected demo

| Condition | Primary 42 exact | Demo 7 exact | Primary clinical | Demo clinical |
| --- | --- | --- | --- | --- |
| Known names /known layouts | 47/50 | 49/50 | 42/44 | 44/44 |
| Known names /new layouts | 5/50 | 5/50 | 31/67 | 26/67 |
| New names /known layouts | 50/50 | 43/50 | 33/33 | 33/33 |
| New names /new layouts | 5/50 | 5/50 | 30/53 | 21/53 |

The primary evaluator covers all 8 architectures across 4 partitions:1,600 model-note evaluations over 200 distinct notes. The separately selected demo adds 200 evaluations of those same notes, not 200 new independent cases. Its development selection was fixed before its final outputs. These conditions test declared patient-name/template novelty, not every kind of identifier novelty. Controls share layouts; dates, doctor/address policies retain their stated limits. Broad supported spelling does not imply generalization. Do not retrain or select a different reported seed after reading these outcomes. Full raw tables include recall, precision, clinical associations, leakage and EOS for every architecture.

Sources: [partitions](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/final_test/summary.json>), [demoFinal](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/demo_final_test/summary.json>), [dataset](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/dataset/manifest.json>), [demo](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/reports/DEMO_SELECTION.json>).

### 25. Week 5 and the experimental method

Saved next-token evaluation on synthetic workflow sequences

| Architecture | Parameters | Accuracy: mean ± SD | Test loss | Macro F 1 |
| --- | --- | --- | --- | --- |
| A: embedding + linear | 1,911 | 69.69% ± 0.77 | 1.547 | 0.407 |
| B: one attention head | 4,239 | 64.06% ± 3.56 | 2.073 | 0.355 |
| C: four attention heads | 4,239 | 62.81% ± 3.62 | 2.234 | 0.326 |
| D: two full blocks | 16,215 | 70.94% ± 1.25 | 1.478 | 0.442 |

Read these as small-sample evidence from the saved Week 5 experiment, not Mitra performance. Across five initialization seeds, the full two-block model D improves mean accuracy by 1.25 percentage points over A. The test set in each saved run contains 64 target tokens across four cases. The evaluated scenario labels include both held-out families and familiar families. The saved JSON does not include a full dataset manifest or histories, so it does not establish a broad reproducibility claim. Mean test loss falls from about 1.547 to 1.478 and macro F 1 rises from about 0.407 to 0.442. The standard deviations are across seeds, not confidence intervals. No statistical significance claim follows from these aggregates. This is a different task and vocabulary. Its token accuracy is not directly comparable to character-token Mitra accuracy. Use it to show the experimental method, not a before/after product metric. The final Mitra tokenizer is fixed task pieces with character fallback. Its complete-note metrics and per-token loss units differ from this older word-token task.

Sources: [results](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/visualizations/arch_results.json>), [runner](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/experiment_runner.py>).

### 26. Live demonstration

Actual checkpoint outputs beside saved benchmark measurements

| View | Action | Interpretation |
| --- | --- | --- |
| Compare models | Select final_seed7/transformer_2 | Demonstrate authored cases and token trace |
| Compare models | Same PHONE note at primary 42 and demo 7 | Retain the primary failure and both labels |
| Inspect heads | Select task_tokens/transformer_2 | Conditional routing and seed 42 intervention |
| Compare components | Read final curves, seed ranges and partitions | State supports and checkpoint policy |

A short demonstration can take six minutes. First show a passing VISITED or Olivia/MRN prompt using the explicitly labelled demo. Then show the phone comparison without repairing either output. Explain that this development-selected candidate is separate from the primary architecture benchmark. For saved head measurements switch to seed 42 so its caption agrees with the underlying experiment. Compare a free-generation intervention and a supplied-prefix attention row as different tests. The offline fallback contains saved actual text and performs no inference. Team members must each trace one prompt independently before rehearsal is complete.

Sources: [ui](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/app.py>), [demo](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/reports/DEMO_SELECTION.json>), [heads](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/investigations/enriched_head_measurements.json>), [partitions](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/final_test/summary.json>).

### 27. What the experiments taught us

A GPT learns parameters that turn a prefix into the next-token distribution

The final demonstration is a real manually assembled GPT, not a deterministic renderer. Broader input coverage and a different token unit improved development behavior compared with earlier experiments, but multiple things changed across those studies. Within the final frozen corpus, compare architecture and seed effects using the same definitions. Do not claim that a head has a universal semantic role or that every added component must improve a small model. Actual final transfer results and failure cases bound the conclusion. Learner understanding requires every presenter to explain the path, not merely show a working interface.

Sources: [protocol](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/protocol.json>), [primary](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/summary.json>), [comparison](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/descriptive_comparison.json>), [heads](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/investigations/enriched_head_measurements.json>), [demoFinal](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/demo_final_test/summary.json>).

### 28. Further validation and learning

Prepared engineering and remaining human steps

| Area | Available now | Next evidence |
| --- | --- | --- |
| Local execution | Real checkpoints and a Gradio interface | Timed team rehearsal and individual trace |
| Cloud training | Guided notebook with source/data hashes | Actual GPU allocation, runtime recovery and CPU compare |
| Model behavior | Development and frozen final measurements | More varied synthetic sources and careful error analysis |
| Later exercises | Scaffolding and fine-tuning are separate work | Choose the method from the task and evidence |

No real Colab session or GPU-to-CPU portability is claimed. The notebook checks actual CUDA allocation, preserves training state to Drive and exports a device reference for CPU comparison. Google/Drive sign-in belongs to Nagar. Local CPU compute was sufficient for this experiment, so no paid compute was purchased. Team roles, exact duration and every presenter explanation remain to be demonstrated. This project uses synthetic notes and is not a validated operational de-identification service. Do not expand this exercise into clinical or compliance claims.

Sources: [cloud](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/train_colab.ipynb>), [task](<C:/Users/Nagar/source/repos/ai-learning-lab/weekly curriculum/06-Tiny-Mitra-Project/TASKS.md>), [demoFinal](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/demo_final_test/summary.json>).

### 29. Appendix: attention with small numbers

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

### 30. Appendix: normalization and the FFN

Operations within a token representation

```text
LN(x) = γ ⊙ (x − μ) / √(σ² + ε) + β
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

LayerNorm uses features within one position, not statistics across all examples in a batch. With x=[1,3], mean is 2 and population variance is 1. Ignoring epsilon for this arithmetic and using gamma=[1,1], beta=[0,0], normalized output is [-1,1]. Learned gamma and beta can rescale and shift it. For the FFN, W1 has shape (64,192), b 1 has shape (192), W2 has shape (192,64), and b 2 has shape (64). The nonlinearity is essential: two linear maps with no nonlinear activation collapse into a single linear map. Our implementation uses GELU while the earlier NumPy study uses ReLU. Avoid claiming expansion itself guarantees richer measured behavior. The formula treats feature vectors as rows. PyTorch nn.Linear stores weight as output-by-input and multiplies by its transpose, so Linear(64,192).weight has shape 192 by 64. This storage convention differs from the mathematical W 1 orientation shown in the formula.

Sources: [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [numpy](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/numpy_transformer_suite.py>), [layernorm](<https://arxiv.org/abs/1607.06450>).

### 31. Appendix: the loss-mask boundary

Excerpt under the final task-piece tokenizer

| Input token | Next target | Score? | Reason |
| --- | --- | --- | --- |
| <INPUT> | PATIENT | No | Source prediction |
| Last source token | <OUTPUT> | No | Protocol boundary |
| <OUTPUT> | PATIENT | Yes | First output token |
| PATIENT | space | Yes | Next output token |
| Last output token | <EOS> | Yes | Learn termination |
| Padding | -100(ignore) | No | Batch filler |

This is an excerpt, not one contiguous six-token example. The final tokenizer gives PATIENT a single ID and preserves spaces as their own token. Shift inputs/targets by one position. At OUTPUT the next target is PATIENT. Ignore source-prediction and padding targets but keep attention access to the source. EOS is scored. -100 is a loss ignore index, not a token ID. Context overflow raises an error so truncation cannot silently remove output/EOS.

Sources: [data](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/data.py>), [tokenizer](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/task_tokenizer.py>), [tests](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/results/mechanics-tests.xml>).

### 32. Appendix: measured parameter count

Full model: vocabulary 146, width 64, FFN 192, context 256

| Component | Calculation | Parameters |
| --- | --- | --- |
| Token embedding | 146 ×64 | 9,344 |
| Position embedding | 256 ×64 | 16,384 |
| Attention, two blocks | 2 ×4 ×(64²+64) | 33,280 |
| FFNs, two blocks | 2 ×(2 ×64 ×192+192+64) | 49,664 |
| All norms | 5 ×(64+64) | 640 |
| Vocabulary head | 64 ×146+146 | 9,490 |
| Total | Sum of six components | 118,802 |

The actual full-model parameter_count is 118,802. Token embeddings and the untied vocabulary projection are different learned parameters. Every linear layer includes bias. Q/K/V and attention output make four 64 to 64 linear maps per block. Each FFN has two linear layers with separate biases. Four block norms plus one final norm contribute 5 times 128 parameters. Context capacity allocates 256 position rows; this does not teach positions absent from training. One versus four heads at fixed 64 width has the same projection parameter count.

Sources: [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [config](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/config.py>), [mitra](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/summary.json>).

### 33. Appendix: the autoregressive loop

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

### 34. Appendix: generalization conditions

Final partitions are separate from validation-only diagnostics

| Partition | Notes | Condition |
| --- | --- | --- |
| Known names / known layouts | 50 | New combinations; familiar pools |
| New names / known layouts | 50 | Separate final-name pool |
| Known names / new layouts | 50 | Final-only structural templates |
| New names / new layouts | 50 | Joint entity/layout transfer |
| Training / validation | 700 / 100 | Validation includes its own novel conditions |

Final partitions contain 50 notes each, with declared entity/template pools. Complete primary evaluation uses the fixed 4,000th-update seed 42 checkpoints, not best-loss selection. The selected demo retains its separately recorded seed and checkpoint. Authored development names, including Olivia, are excluded from final-test names. Validation has its own unfamiliar pools. No-identifier controls share layouts and some non-patient identifier patterns recur; novelty is specific to the declared condition. All source strings are unique. The vocabulary comes from schema/ASCII declarations, not fitting on final text.

Sources: [dataset](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/dataset/manifest.json>), [protocol](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/protocol.json>), [partitions](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/final_test/summary.json>), [demoFinal](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/demo_final_test/summary.json>).

### 35. Appendix: questions for the technical discussion



Architecture changes capacity and optimization as well as the intended feature. Equal update counts are not equal compute; most component comparisons have one seed. The 24 diagnostic validation notes use known pools, while complete final partitions test declared new names/layouts. The primary architecture benchmark stays seed 42 at 4,000 updates. A separately labelled candidate passes the declared development gate, without using final outcomes to choose its seed. A supplied prefix is not free generation. Head ablations measure checkpoint dependence on a small sample, not universal head meaning. Gradio only loads weights and calls the same generation function. Every presenter should answer these questions after teaching and practice.

Sources: [comparison](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/descriptive_comparison.json>), [demo](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/reports/DEMO_SELECTION.json>), [dataset](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/dataset/manifest.json>), [heads](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/investigations/enriched_head_measurements.json>), [ui](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/app.py>).

### 36. Appendix: evidence and references



Assignment and local code/evidence sources are in relevant slide notes. Architecture uses explicit scaled dot-product attention and learned positions with pre-normalization blocks. See Vaswani et al. https://arxiv.org/abs / 1706.03762 and Ba et al. https://arxiv.org/abs / 1607.06450 for original mechanisms. Attention-interpretation discussions include https://aclanthology.org/N 19-1357/ and https://aclanthology.org/D 19-1002/. Current numerical claims come from final primary/repeat/raw generation evidence, not those external papers. Week 5 results belong to their separate task. Actual cloud execution and individual mastery are not claimed.

Sources: [assignment](<C:/Users/Nagar/source/repos/ai-learning-lab/topics/Mitra Topics/Week 6 — Build Your Own Tiny-GPT_Excercise.md>), [model](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/model.py>), [protocol](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/protocol.json>), [partitions](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/final_test/summary.json>), [demoFinal](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/Week 6/tiny-mitra/artifacts/mitra_final/results/demo_final_test/summary.json>), [results](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/visualizations/arch_results.json>), [attention](<https://arxiv.org/abs/1706.03762>), [layernorm](<https://arxiv.org/abs/1607.06450>), [interpretation](<https://aclanthology.org/N19-1357/>), [interpretation2](<https://aclanthology.org/D19-1002/>).


## Timing and team preparation

The timings below are proposed routes, not a completed rehearsal. Every presenter still needs to trace a prompt independently. Assign names together rather than assuming one person owns all technical explanations.

| Slot | Suggested route | Questions and demo |
| --- | --- | --- |
| 30 minutes | Slides1–3,5–13,16–18,21,23–24,27–28; summarize the rest from notes | Reserve six minutes for a short demo/questions; keep appendix for questions |
| 45 minutes | All28 core slides at roughly one minute each, with more time for Q/K/V, output loss and actual findings | Reserve six minutes for live generation/head evidence and about ten minutes for questions |
| 60 minutes | All core slides plus selected arithmetic/normalization/checkpoint appendix explanations | Reserve time for each presenter to explain one prompt and for audience questions |

Before speaking: open the actual deck in the presentation application, check projector readability, start the local demo, visually exercise the new incremental-token/test-metric controls and keep the recorded-output fallback available. The latest automated browser recheck was blocked after a connection-refused page; API checks pass, but that does not replace this visual check. No native PowerPoint application test or timed team rehearsal has been claimed. The copied guide's local source links assume the original repository; the demo archive includes project code/evidence but excludes private reference books.
