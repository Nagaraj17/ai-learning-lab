# Tiny Mitra: presentation and learning preparation

Presentation: **Friday, October 23, 2026**. Audience: CTO, CEO, COO, technical architect, Director of Engineering, and other leaders. Available speaking time: **30–60 minutes**.

This edition records the project state on October 17. It provides the explanation, technical questions and finish plan. **Mitra has no verified training results yet.** The sibling starter package contains code, but it has not been established as the learner's completed, understood and validated implementation. Week 5 results are separate evidence and must never be labeled Mitra results.

## The story to tell

“We learned how a neural network turns text into a next-token prediction, built the components from first principles, and used experiments to question whether added complexity actually helped. Mitra brings those components together in a small conditional-generation task. We evaluate whether it removes synthetic identifiers while preserving the specified content, including on held-out examples.”

The final version of that statement must include what the trained Mitra actually does. A good talk can explain a limited result. It cannot substitute an expected target for a model generation.

## Speaking routes

The deck has **24 core slides and 8 appendix slides**. The notes carry detail so the visible slides do not become a textbook.

| Segment | Slides | 30-minute route | 60-minute route |
| --- | --- | --- | --- |
| Problem, learning journey and evidence status | 1–4 | 4 min | 5 min |
| Task framing, tokenization and context | 5–9 | 5 min | 8 min |
| Attention, blocks and the token journey | 10–14 | 6 min | 10 min |
| Learning, generation and baselines | 15–17 | 4 min | 6 min |
| Week 5 evidence and Mitra evaluation | 18–21 | 4 min | 7 min |
| Actual demo, evidence status and conclusion | 22–24 | 5 min | 7 min |
| Selected appendix explanation | 25–32 | Use only for questions | 7 min |
| Discussion and timing buffer | As needed | 2 min | 10 min |
| Total | | 30 min | 60 min |

For the final presentation, slide 23 should report what was completed and what remains limited. Keep an honest preparation status if the work is still unfinished. The current slide 22 describes the planned demonstration and contains no actual successful model output.

## A realistic finish plan

Reserve one focused learning block each day and use longer weekend blocks for implementation work. The sequence matters more than expanding the feature list. The items below are proposed work sessions, not scheduled reminders.

| Date | Focus | Concrete stopping point |
| --- | --- | --- |
| Sat Oct 17 | Recover the full token journey and inspect the starter | Explain one prompt, target shift, masks and output distribution without hiding behind library names |
| Sun Oct 18 | Data and correctness | Explicit split manifest, duplicate and ID checks, tokenizer coverage report, mask checks, one tiny batch that can be learned |
| Mon Oct 19 | Train the architecture ladder | A, B and C checkpoints, saved configurations and validation-selected checkpoints on the same split |
| Tue Oct 20 | Evaluate and investigate | Generated outputs, per-partition task metrics, matched context pair, controlled temperature comparison, one analyzed failure |
| Wed Oct 21 | Test the attention interpretation and complete the demo | Capture the prediction-specific attention row, run a counterfactual, record the actual demonstration, close critical gaps |
| Thu Oct 22 | Finish and rehearse | Replace preparation claims with measured evidence, freeze the checkpoint, rehearse the 30-minute route, practice the technical questions |
| Fri Oct 23 | Present the frozen evidence | Verify local demo and video playback, then use the prepared presentation |

If work pressure reduces available time, prioritize the required A/B/C comparison, a sound evaluation, one context test, the required temperature and failure experiments, and the demonstration. Optional depth sweeps and tokenizer comparisons come after that. If an optional experiment is absent, state that it is an extension.

## The first refresher: one exact token journey

Consider this synthetic source and authored target:

```text
Source: PATIENT John Smith DIAGNOSIS NSCLC
Target: PATIENT [NAME] DIAGNOSIS NSCLC
```

The serialized training example is:

```text
<BOS> <INPUT> PATIENT John Smith DIAGNOSIS NSCLC <OUTPUT> PATIENT [NAME] DIAGNOSIS NSCLC <EOS>
```

1. **Tokenization:** Each marker, word and placeholder becomes a vocabulary ID. For this word-level example, the prefix through `<OUTPUT>` has 8 tokens. The IDs are table addresses, not quantities.
2. **Embeddings:** A lookup returns 64 features for each token under the starter defaults. Adding its position embedding gives shape `(1, 8, 64)`.
3. **Attention projections:** Each learned Q/K/V projection transforms those representations. Four heads reorganize features into `(1, 4, 8, 16)`. Every head keeps all eight token positions.
4. **Compatibility and masking:** Q times transposed K gives `(1, 4, 8, 8)` scores. Division by `sqrt(16) = 4` scales each score. Causal and padding masks remove ineligible keys before softmax.
5. **Value mixing:** Each attention row weights value vectors. Merging heads and applying the output projection restores `(1, 8, 64)`.
6. **Decoder processing:** Pre-normalization, residual additions and the FFN repeat through two blocks. The FFN changes feature width `64 → 192 → 64` internally. The outer shape stays `(1, 8, 64)`.
7. **Vocabulary projection:** A final normalization and linear head produce `(1, 8, V)`, where `V` is the actual training vocabulary size.
8. **First generated token:** The last position is `<OUTPUT>`. Its logits predict the next token, whose target is `PATIENT`. It does not immediately predict `[NAME]`.
9. **Next iteration:** Append `PATIENT`. The enlarged prefix has 9 tokens. Its final-position logits now have `[NAME]` as the intended next target.
10. **Completion:** Repeat for `DIAGNOSIS`, `NSCLC` and `<EOS>`. Only the generated continuation is the transformed note. The original source remains present as context.

These are intended targets and illustrative shapes. Until a trained checkpoint produces them, they are not observed generations.

### Why each part exists

| Problem | Mechanism | What it still cannot guarantee |
| --- | --- | --- |
| Integer IDs have no useful geometry | Trainable embeddings | Useful features without sufficient training |
| A sequence needs order | Position representations | Recovery of spelling lost by `<UNK>` |
| The next output may depend on an earlier source fact | Causal self-attention | That training actually learned the right dependency |
| One distribution may be restrictive | Multiple learned heads | Distinct meaningful roles for every head |
| Mixed features need nonlinear transformation | FFN with an activation | Improved held-out performance on every task |
| Repeated transformations need manageable signal paths | Residual additions and LayerNorm | Stability under every learning rate and initialization |
| Hidden vectors need to become token scores | Vocabulary projection | Generation of a word absent from the output vocabulary |
| A training example needs a learning signal | Output-only next-token cross-entropy | Reliable full-sequence generation from low loss alone |

### Where the loss and gradients go

The first scored prediction occurs at `<OUTPUT>` and targets `PATIENT`. At the next position, the target is `[NAME]`. The loss mask also scores the final `<EOS>`. It excludes source-prediction targets and padding.

For one correct-token probability `p`, the loss contribution is `−ln(p)`. A probability of `0.1` gives about `2.303`, while `0.8` gives about `0.223`. The smaller number rewards greater probability on the intended token. The natural logarithm converts products of sequence probabilities into sums of token contributions.

Backpropagation follows every dependency that contributed to those output losses. Source embeddings can receive gradients through attention even though source-prediction targets have loss weight zero. A mask can remove direct loss at a position without removing that position from the computation graph.

### Read before attempting a teach-back

After reading the example, use the first prompt below as the next one-at-a-time learning checkpoint. The remaining prompts are available for later practice, not a cold assessment.

1. Why is the first target after `<OUTPUT>` `PATIENT` rather than `[NAME]`?
2. After appending `PATIENT`, which tensor position predicts `[NAME]`?
3. Why can the source still receive gradients when its direct loss is masked?
4. What information becomes impossible to recover if two names both become `<UNK>`?
5. What would demonstrate that the model actually used an earlier source diagnosis?

**My understanding:** Leave this space for your own explanation. Do not mark the concept mastered merely because the notes or code exist.

## Questions likely to matter to the CTO

**Why use a GPT for a task that rules or NER can solve?**

The assignment tests understanding of decoder-only conditional generation. Tightly structured synthetic notes may favor deterministic rules. A token-classification model with a replacement stage is also a valid de-identification design. We should compare those approaches before claiming application superiority. A decoder introduces copying, hallucination and evaluation challenges that are part of the learning exercise.

**What did the Transformer actually add?**

Mechanistically, attention makes earlier source representations available to a later output prediction. Empirically, the A/B/C comparison and matched context pair must show whether training used that capability. The existing Week 5 experiment shows a small gain on a small test. It does not answer Mitra's result question.

**Does zero leakage on unseen names prove de-identification?**

No. If a word-level output vocabulary excludes the names, the decoder cannot generate them literally. A zero rate may follow from representation limits. Test known identifiers, inspect unknown-token rates, preserve raw source annotations, and assess copying of allowed content. A finite synthetic test still cannot establish deployment safety.

**What is the difference between the causal mask and output loss mask?**

The causal mask changes which keys a query can read. The loss mask changes which next-token predictions contribute to the optimization objective. The padding mask separately prevents filler keys from contributing. Test these properties independently.

**Are the head roles interpretable?**

Each head has independent learned projections, so differing behavior can emerge. A heatmap alone cannot assign a reliable business role. Inspect the exact query row for the decision, propose an interpretation, then test it with an input intervention or ablation.

**Why could a model without attention get many tokens right?**

Teacher forcing supplies the correct previous output token. Local output grammar and position can predict frequent fixed tokens. That creates respectable average token accuracy while the model remains unable to retrieve a changed clinical fact from the source at a matched output prefix. Full generations and controlled paired tests expose that gap.

**What does from scratch mean here?**

Week 5 implements the learning machinery in NumPy, including manual backward passes. The Mitra starter uses PyTorch tensor operations, basic layers and automatic differentiation while explicitly implementing Q/K/V attention, masking, head splitting and the decoder block. It does not use a pretrained GPT, `nn.Transformer` or `nn.MultiheadAttention`. State that boundary precisely and show the code you understand.

**Can temperature make the output correct?**

Temperature rescales logits before sampling. It adds no knowledge. Low temperature concentrates existing preferences and can confidently repeat an error. Greedy argmax is unchanged by positive rescaling. Keep decoding method fixed when measuring temperature.

**What would make a result reproducible?**

Save the exact generated dataset and split manifest, vocabulary, model configuration, initialization and sampling seeds, training command, environment versions, validation-selected checkpoint, evaluation prompts and raw generated outputs. Saving only an aggregate result table is insufficient.

**What would need to change before this became an application?**

A wider and independently evaluated data distribution, a representation that preserves required text, more complete identifier and content checks, robust handling of long inputs, controlled generation, operational review and a clear failure-handling process. Training on 500–1,000 synthetic examples provides an educational experiment, not evidence of production readiness.

## Specific starter findings to resolve before trusting measurements

The inspected package lives at [the sibling starter](</C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra>). This preparation pass did not change it, train it, or validate it with tests.

| File | Inspected behavior | Why it matters |
| --- | --- | --- |
| `data.py` | Test patient and doctor names are held out, but all four templates occur across splits | It does not establish unseen-template generalization |
| `data.py` | MRNs and phones come from random number generation | Randomness does not guarantee disjoint identifiers |
| `tokenizer.py` | A fixed word vocabulary maps new words to `<UNK>` | Spelling loss and inability to emit unseen words affect interpretation |
| `metrics.py` | Placeholder scoring intersects type counts | A placeholder in the wrong field can receive credit |
| `metrics.py` | Clinical preservation checks marker presence | It does not verify relationships, duplicates, negation or extra invented content |
| `generation.py` | `temperature <= 0.15` selects argmax, higher values sample | The default temperature comparison changes two factors |
| `data.py` and `generation.py` | Training slices sequences and generation keeps a moving window | Truncation can remove the target, EOS or source context |
| `train.py` | Evaluation averages batch loss means | With different target counts, aggregate summed token loss and scored-token counts for a token-weighted loss |

Do not treat these notes as a substitute for an implementation review and executed tests. `train.py` and the existing tests were inspected, but they were not executed in this preparation pass. Inspect and validate the end-to-end UI before making broader correctness claims.

## The evidence the final talk needs

- A/B/C result table with actual parameter counts, training budget, train/validation/test definitions, and checkpoint-selection rule.
- Teacher-forced continuation loss and token accuracy, separated from complete generated-sequence evaluation.
- Full-sequence exact match, annotated identifier leakage, aligned placeholder precision/recall, content preservation and termination, with sample counts.
- Results partitioned into familiar examples, unseen identifiers, unseen templates and combined holdouts where implemented.
- One matched context experiment where the final output prefix is identical and an earlier source fact changes.
- Temperature experiments with fixed sampling mode and controlled random seeds, plus a separate greedy baseline.
- One inspected prediction-specific attention row and an intervention that supports or challenges the interpretation.
- At least one failure case with a specific diagnosis and a recorded demonstration of the frozen checkpoint.

The current deck contains real Week 5 measurements and clearly labeled Mitra teaching examples. It intentionally does not invent a Mitra result table. Insert measured Mitra evidence after training and evaluation, then rehearse the updated slide count.

## Rehearsal and demo procedure

1. Explain the problem and one input/target pair without technical vocabulary first.
2. Trace the same pair through the model with tensor shapes.
3. Explain why training loss and generation differ.
4. Present the baseline comparison and a controlled context test.
5. Show one success and one failure from the frozen checkpoint.
6. State the limits of the data, tokenizer and test size.
7. Close by answering what generalized, what memorized and what remains unproven.

For the live demo, keep the checkpoint and tokenizer local and open the exact command before the meeting. Retain saved actual outputs and a short recording. If the live run fails, switch openly to the recording or saved trace and explain the failure. Never display an authored expected target as if the model generated it.

## Evidence notes for the saved Week 5 benchmark

The saved JSON contains **8 variants × 5 initialization seeds = 40 runs**. Each inspected run includes **64 scored target tokens** across four cases, with a mixture of familiar and held-out scenario labels. These are initialization repeats on one split, not five independent dataset splits. The table reports across-seed standard deviations, not confidence intervals.

The two-block full model's mean accuracy is **70.9375%** versus **69.6875%** for the embedding-only model, a difference of **1.25 percentage points**. Its macro F1 is approximately **0.442** versus **0.407**. The saved test-loss means are approximately **1.478** versus **1.547**. The no-residual variant has a slightly lower mean loss than the full model but lower accuracy. The no-FFN implementation also omits the normalization associated with that sublayer, so it is not a clean single-component isolation.

The saved artifact does not include the full training histories or a complete dataset manifest. This preparation work recalculated the reported means from the saved individual-run metrics. It did not rerun Week 5 training or establish statistical significance.


## Slide-by-slide explanation and speaking notes

### 1. Tiny Mitra

Understanding a GPT through
synthetic note de-identification

This presentation connects our first-principles learning to a specific engineering question: can a small causal language model transform synthetic identifying text while preserving the intended clinical markers? Explain the learning objective before discussing architecture. The current edition documents completed Week 5 evidence and the Mitra design. Mitra training, evaluation and a verified demonstration remain pending. This is a preparation edition, not a report of completed Mitra experiments.

Sources: [assignment](<C:/Users/Nagar/source/repos/ai-learning-lab/topics/Mitra Topics/Week 6 — Build Your Own Tiny-GPT_Excercise.md>), [results](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/visualizations/arch_results.json>).

### 2. Our learning journey

Each week removed a limitation in the previous model

| Stage | What we built or studied | Why it matters for Mitra |
| --- | --- | --- |
| Week 1 | Forward pass, loss and learning | Turn prediction errors into parameter updates |
| Week 2 | Tokenization and embeddings | Represent text as trainable vectors |
| Week 3 | Causal self-attention | Use relevant earlier context |
| Week 4 | Multiple attention heads | Allow several context distributions |
| Week 5 | Transformer blocks and ablations | Test what added components actually buy |

Use this slide as a narrative, not as a list of technologies. A forward pass computes scores, but loss and backpropagation make those scores learnable. Embeddings replace sparse token identities with learned vectors. Attention lets each position gather information from earlier positions. Multiple heads provide separate learned projections. A complete block combines attention with feature transformation, normalization and residual paths. The tracker marks Weeks 1 to 4 complete and Week 5 in progress. That is a record of learning progress, not a fresh assessment of personal mastery. The Week 5 repository contains executable NumPy components and saved experiment results.

Sources: [progress](<C:/Users/Nagar/source/repos/ai-learning-lab/PROGRESS.md>), [numpy](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/numpy_transformer_suite.py>), [results](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/visualizations/arch_results.json>).

### 3. Mitra’s transformation task

Synthetic examples with an explicit redaction policy

```text
PATIENT John Smith MRN 482991
DIAGNOSIS NSCLC BIOMARKER EGFR

PATIENT [NAME] MRN [MRN]
DIAGNOSIS NSCLC BIOMARKER EGFR
```

Read the source and target aloud. The target removes the synthetic patient name and record number while retaining the diagnosis and biomarker. These are task rules for a controlled educational dataset. We are not making a legal compliance claim or validating a clinical product. State clearly that the displayed target is authored ground truth, not an observed model generation. A deterministic rule system could solve tightly structured examples. We build a decoder-only model because the assignment asks us to understand conditional generation. An eventual application comparison should include rules and a token-classification pipeline.

Sources: [assignment](<C:/Users/Nagar/source/repos/ai-learning-lab/topics/Mitra Topics/Week 6 — Build Your Own Tiny-GPT_Excercise.md>), [data](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/data.py>).

### 4. What we can claim today

Completed evidence and intended Mitra work have different status

| Area | Current evidence | Claim boundary |
| --- | --- | --- |
| Week 5 | Saved runs across 8 architecture variants | Small evaluation supports limited comparisons |
| Mitra learning | Assignment and concept notes available | Architecture and objective can be explained |
| Mitra code | Separate starter package located | Code availability does not establish correctness |
| Mitra results | No verified training or evaluation yet | Generalization and demonstration remain unproven |

This is an important credibility slide for a technical audience. We found a separate starter package, but the learner confirmed Mitra has not yet been implemented as completed project work. Treat that package as unvalidated starter material. No checkpoint or result artifact was found in the inspected starter tree. Do not call its expected outputs successful generations. Before presenting this as a completed experiment, replace the status with the evidence actually produced. There are eight variants in the saved Week 5 JSON even though the progress tracker still refers to a seven-model benchmark.

Sources: [results](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/visualizations/arch_results.json>), [progress](<C:/Users/Nagar/source/repos/ai-learning-lab/PROGRESS.md>), [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>).

### 5. A transformation as next-token prediction

The same sequence contains the source and the desired continuation

The model predicts a next-token distribution at every input position. We serialize each example as source text followed by a boundary token and the desired output. During training, the position occupied by <OUTPUT> predicts PATIENT. The next position, occupied by PATIENT, predicts [NAME]. Later, DIAGNOSIS predicts the appropriate clinical value using the earlier source. <BOS>, <INPUT>, <OUTPUT> and <EOS> are vocabulary entries with learned embeddings. Their names are conventions, not hard-coded reasoning. A model must learn how those boundaries relate to the training task. The loss mask scores the output continuation and EOS while retaining the source as context.

Sources: [data](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/data.py>), [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>).

### 6. Training and generation

Parallel scoring during training, one appended token during inference

| Aspect | Training | Generation |
| --- | --- | --- |
| Available text | Source and correct output prefix | Source and model-generated prefix |
| Model input | Shifted complete example | Current prompt plus generated tokens |
| Prediction used | All scored output positions | Final position only |
| Update | Loss, backpropagation, optimizer step | Append one token without weight updates |
| Main risk | Target leakage through an incorrect mask | An early error becomes future context |

Teacher forcing means the training input contains the correct previous output token. A causal mask prevents a position from reading a later target, so these position-wise predictions can be computed together. Autoregressive generation has no gold output. We take logits from the last supplied token position, choose one next token, append it, and repeat. The distinction explains why a low teacher-forced loss can coexist with poor full-sequence generation: generation conditions on its own mistakes. The causal attention mask, padding mask and loss mask solve different problems. We must test all of them rather than relying on a plausible diagram.

Sources: [data](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/data.py>), [generation](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/generation.py>), [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>).

### 7. Tokenization defines what the model can see



The starter uses a regex word tokenizer and fits it on training sequences only. This avoids building the vocabulary from test examples, but it creates an important limitation. Olivia and Noah can both become the same <UNK> token. Adding different position vectors does not recover their lost spelling. Redacting a name after PATIENT may still work from context, but that result would not prove character-level understanding of names. An unseen diagnosis might also become <UNK>, and a fixed output vocabulary cannot generate the original unseen word. Worse, the decoder cannot emit a test-only name absent from its vocabulary, so zero literal leakage on such names can be structurally guaranteed. Test known identifiers too and distinguish tokenizer coverage from model behavior.

Sources: [tokenizer](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/tokenizer.py>), [book](<C:/Users/Nagar/source/repos/ai-learning-lab/resources/references/Hands-on- Large Language Models.md>), [data](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/data.py>).

### 8. Embeddings and position

A token ID selects a vector, then position adds order information

```text
X[b, t, :] = E[token_id] + P[t]
```

An ID is an address, not a meaningful numerical quantity. A token with ID 40 is not twice the meaning of ID 20. The embedding table has one row per vocabulary entry and one column per feature dimension. Lookup retrieves that row. The position table adds another vector of the same width so order is available to the network. These dimensions are learned features and should not be assigned fixed human meanings. The tiny two-dimensional arithmetic on the slide only illustrates addition. It is not a vector measured from the project. Where used in AI: every subsequent Q/K/V projection transforms these combined input representations.

Sources: [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>), [config](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/config.py>), [book](<C:/Users/Nagar/source/repos/ai-learning-lab/resources/references/Hands-on- Large Language Models.md>).

### 9. A context test with the same output prefix

Only one earlier source token changes

```text
Source A: PATIENT John DIAGNOSIS NSCLC
Source B: PATIENT John DIAGNOSIS ASTHMA

Shared generated prefix: PATIENT [NAME] DIAGNOSIS
Expected next token: NSCLC for A, ASTHMA for B
```

This pair avoids a weak context demonstration in which a changed immediately preceding output label already determines the answer. At the decision point shown here, the latest output token is DIAGNOSIS in both prompts, and the sequence lengths match. Only the earlier source diagnosis changes. Model A sees only the current token and position, so its logits at this point cannot distinguish the two sources. An attention model can use the earlier source. That gives a concrete capability difference, but success still requires an executed test. Include these cases inside the supported task distribution and reserve separate unseen-template tests. Show both next-token probabilities and complete generations.

Sources: [assignment](<C:/Users/Nagar/source/repos/ai-learning-lab/topics/Mitra Topics/Week 6 — Build Your Own Tiny-GPT_Excercise.md>), [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>).

### 10. Attention retrieves a weighted mixture

Learned compatibility determines how much each earlier value contributes

```text
A = softmax((QKᵀ / √dₕ) + M)     H = AV
```

Q, K and V are different projections of the current hidden representations. A query at one position has a dot product with every eligible key. Transposing K aligns its feature dimension for those comparisons. Dividing by the square root of head width limits score scale under the standard variance intuition. Softmax exponentiates each eligible score and divides by the row sum, yielding nonnegative weights that sum to one. Multiplying by V gives a weighted mixture, not a discrete database lookup. In a causal layer, later keys are ineligible. The query at a generation step belongs to the final prefix position, which predicts the next token. Calling a head a name-detector requires measured evidence.

Sources: [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>), [attention](<https://arxiv.org/abs/1706.03762>), [interpretation](<https://aclanthology.org/N19-1357/>), [interpretation2](<https://aclanthology.org/D19-1002/>).

### 11. Three masks, three jobs

Causal attention prevents future targets from becoming visible

| Query / key | <OUTPUT> | PATIENT | [NAME] | DIAGNOSIS |
| --- | --- | --- | --- | --- |
| <OUTPUT> | allow | block | block | block |
| PATIENT | allow | allow | block | block |
| [NAME] | allow | allow | allow | block |
| DIAGNOSIS | allow | allow | allow | allow |

This table shows an excerpt of the causal pattern over output positions. Every output query can also read eligible earlier source positions, which are omitted from this small matrix. Blocked attention scores become negative infinity before softmax, so their attention probabilities become zero. The diagonal remains visible because a token representation predicts the next token. Padding keys are masked separately. The output-only loss mask decides which predictions contribute to the objective and does not control attention visibility. A useful causal test changes a future token and verifies that earlier logits stay unchanged in evaluation mode. A separate test changes padded input values and checks that nonpadded predictions do not change.

Sources: [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>), [data](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/data.py>).

### 12. Multiple heads examine the full sequence

The feature dimension divides across heads, while every head keeps all token positions

| Operation | Illustrative shape | Meaning |
| --- | --- | --- |
| Input | (B, T, 64) | 64 features at every position |
| Q, K, V after split | (B, 4, T, 16) | Four different learned projections |
| Per-head attention | (B, 4, T, T) | A separate distribution per query and head |
| Concatenate contexts | (B, T, 64) | Rejoin 4 × 16 features |
| Output projection | (B, T, 64) | Mix information across head outputs |

The starter default has d_model 64 and four heads, so d_head is 16. The head split reorganizes projected features. It does not divide the sequence into four sections, and head one is not automatically assigned names while head two handles record numbers. Each head gets every eligible token position. Concatenation joins feature channels. The learned output projection can combine these channels. Holding d_model fixed keeps the standard attention projection parameter count approximately constant as the number of heads changes. More heads do not automatically mean more parameters or better results. We must measure whether distinct useful behavior develops.

Sources: [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>), [config](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/config.py>), [attention](<https://arxiv.org/abs/1706.03762>).

### 13. A complete decoder block

The starter uses normalization before each sublayer

```text
U = X + Attention(LN₁(X))
Y = U + FFN(LN₂(U))
```

This is a pre-LayerNorm block, matching both the inspected Week 5 NumPy class and the Mitra starter. The older Week 5 submission prose displays a different ordering, so use the code when explaining the implementation. LN1 transforms X before attention, then the residual adds the attention update to X. LN2 transforms U before the feed-forward network, followed by another residual addition. LN computes feature mean and variance for each position, then applies learned scale and offset. The FFN expands from 64 to 192 features, applies GELU in this starter, and projects back to 64. Neither component alone guarantees generalization or training stability under arbitrary settings.

Sources: [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>), [numpy](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/numpy_transformer_suite.py>), [layernorm](<https://arxiv.org/abs/1607.06450>).

### 14. The complete token journey

Illustrative forward pass with one 8-token prefix and four heads

| Stage | Shape | What changes |
| --- | --- | --- |
| IDs | (1, 8) | Text becomes vocabulary indices |
| Token + position | (1, 8, 64) | Each position receives a learned vector |
| Q / K / V | (1, 4, 8, 16) | Features reorganize across heads |
| Attention weights | (1, 4, 8, 8) | Each query weights earlier keys |
| Two decoder blocks | (1, 8, 64) | Context and features evolve |
| Vocabulary logits | (1, 8, V) | Each position scores every vocabulary entry |
| Final position | (1, V) | One distribution predicts the next token |

Use the eight-token prefix <BOS> <INPUT> PATIENT John Smith DIAGNOSIS NSCLC <OUTPUT>. The prefix length is eight, but vocabulary size V depends on the tokenizer fitted to the training set. The table uses the starter default model width and head count, not measured activations. At <OUTPUT>, the first desired generated token is PATIENT. The model does not immediately emit [NAME]. After PATIENT is appended, the next forward pass predicts the placeholder. Both blocks preserve the outer batch, time and model-width shape. The final linear layer changes only the feature dimension from 64 to V. The generation loop selects the last time position because it predicts the continuation after the supplied prefix.

Sources: [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>), [generation](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/generation.py>), [config](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/config.py>).

### 15. Learning the output continuation

The source contributes context while selected targets contribute loss

```text
L = −Σₜ mₜ log p(yₜ | prefixₜ) / Σₜ mₜ
```

The minus sign makes negative log probabilities into positive penalties. A larger correct-token probability reduces the penalty. We divide by the number of scored targets so padding and prompt length do not dilute the task loss. During backpropagation, gradients still flow through source representations because the output predictions depend on them through attention. The mask removes direct source-token prediction loss, not source learning. The optimizer uses gradients to update parameters. For a simple SGD intuition, a parameter becomes its old value minus learning rate times its loss gradient. The starter can use autograd for differentiation while manually expressing attention. Include EOS in the scored continuation and verify off-by-one alignment with a short example.

Sources: [data](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/data.py>), [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>), [mathbook](<C:/Users/Nagar/source/repos/ai-learning-lab/resources/references/Deep Learning.md>).

### 16. Generation and temperature



Temperature T must be positive in this formula. Greedy decoding explicitly selects the largest logit. Dividing logits by any positive temperature preserves their ordering, so greedy token choice does not change. Sampling draws from the temperature-adjusted probabilities, which can change the selected token. A low temperature can still sample a lower-ranked token and is not a correctness guarantee. The starter currently uses greedy decoding at temperatures up to 0.15 and sampling above that threshold. Comparing its settings 0.1, 0.7 and 1.2 therefore confounds temperature with decoding method. Separate those controls before the experiment. Log EOS termination and context overflow, and avoid silently dropping the source.

Sources: [generation](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/generation.py>).

### 17. The Mitra baseline comparison

Planned experiments on the same frozen data split

| Model | Architecture | Question |
| --- | --- | --- |
| A | Token + position embeddings, linear head | How far can local output patterns go? |
| B | One causal attention head with residual | What does access to source context add? |
| C | Two complete decoder blocks | What does the full small GPT add? |
| Rules | Deterministic structured redaction | How does the learning model compare on this task? |

A, B and C satisfy the required architecture ladder. Model B in the starter adds a residual around the attention layer, which should be disclosed. This is not a perfectly isolated component experiment because depth, parameter count, normalization and FFN capacity differ. Use the same tokenizer, split, continuation loss, evaluation prompts and decoding settings. State whether training budgets match epochs, update steps, tokens or elapsed time. Select checkpoints using validation data and evaluate the frozen test once for the final report. A deterministic baseline is an application comparator, not a replacement for the required GPT. Prioritize A/B/C before optional one-block or four-block experiments given the deadline.

Sources: [assignment](<C:/Users/Nagar/source/repos/ai-learning-lab/topics/Mitra Topics/Week 6 — Build Your Own Tiny-GPT_Excercise.md>), [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>).

### 18. Week 5: a modest measured improvement

Saved next-token evaluation on synthetic workflow sequences

| Architecture | Parameters | Accuracy: mean ± SD | Test loss | Macro F1 |
| --- | --- | --- | --- | --- |
| A: embedding + linear | 1,911 | 69.69% ± 0.77 | 1.547 | 0.407 |
| B: one attention head | 4,239 | 64.06% ± 3.56 | 2.073 | 0.355 |
| C: four attention heads | 4,239 | 62.81% ± 3.62 | 2.234 | 0.326 |
| D: two full blocks | 16,215 | 70.94% ± 1.25 | 1.478 | 0.442 |

Read these as small-sample evidence from the saved Week 5 experiment, not Mitra performance. Across five initialization seeds, the full two-block model D improves mean accuracy by 1.25 percentage points over A. The test set in each saved run contains 64 target tokens across four cases. The evaluated scenario labels include both held-out families and familiar families. The saved JSON does not include a full dataset manifest or histories, so it does not establish a broad reproducibility claim. Mean test loss falls from about 1.547 to 1.478 and macro F1 rises from about 0.407 to 0.442. The standard deviations are across seeds, not confidence intervals. No statistical significance claim follows from these aggregates.

Sources: [results](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/visualizations/arch_results.json>), [runner](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/experiment_runner.py>).

### 19. Week 5: component trade-offs

The metric chosen changes the comparison

| Variant | Test accuracy | Test loss | Observed limitation |
| --- | --- | --- | --- |
| Full model | 70.94% | 1.478 | Reference comparison |
| Without FFN | 70.94% | 1.694 | Same accuracy, worse loss |
| Without LayerNorm | 68.13% | 1.816 | Lower accuracy, worse loss |
| Without residuals | 67.81% | 1.473 | Lower accuracy, slightly lower loss |

The no-FFN variant matches the full model accuracy at 70.94 percent while its mean test loss is worse. That illustrates why accuracy alone can hide changes in predicted probabilities. The no-residual variant has slightly lower mean test loss than the full model but lower accuracy. Do not claim the full model wins every metric. The no-LayerNorm variant performs worse on these saved metrics, but this does not establish that every Transformer without normalization diverges. These are architecture changes within one experimental setup. In the implementation, removing FFN also omits its associated normalization, so this ablation cannot isolate FFN perfectly. Report the observed differences and limits together.

Sources: [results](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/visualizations/arch_results.json>), [numpy](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/numpy_transformer_suite.py>).

### 20. Generalization requires distinct test conditions

Names, layouts and vocabulary coverage test different capabilities

| Test partition | What changes | What it can establish |
| --- | --- | --- |
| Familiar distribution | New examples from known templates | Basic task fit and complete generation |
| Unseen identifiers | Held-out names and exact IDs | Behavior beyond training identifiers |
| Unseen templates | Held-out field order or sentence pattern | Transfer beyond a memorized layout |
| Combined holdout | New identifiers and a new template | A harder joint generalization check |

The starter holds out patient and doctor names for test, but shares its four templates across all splits. Validation uses the training name pool. Random record-number and phone generation does not formally guarantee disjoint IDs. A new sentence generated from a familiar template is not unseen-template evaluation. Partition identifiers and template families explicitly, detect duplicate source-target pairs, and save the data manifest. Use validation conditions resembling the intended tests for model selection without repeatedly tuning on final test examples. For a word tokenizer, report unknown-token rates separately for identifiers and preserved clinical tokens. These distinctions prevent a vocabulary artifact from being described as semantic generalization.

Sources: [data](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/data.py>), [tokenizer](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/tokenizer.py>).

### 21. Mitra needs task-specific evaluation

Teacher-forced token accuracy alone cannot describe generated redaction quality

| Metric | Definition | Failure it reveals |
| --- | --- | --- |
| Exact match | Entire generated target equals reference | Missing, extra or reordered content |
| Identifier leakage | Generated outputs retaining a source identifier / outputs | Under-redaction, even beside a placeholder |
| Aligned placeholder P / R | Correct replacements / generated or required replacements | Wrong type, wrong location, over-redaction |
| Clinical preservation | Required clinical facts retained / required facts | Lost or altered content |
| Termination and validity | EOS, length limits, malformed outputs | Incomplete or uncontrolled generation |

Define the unit of analysis before reporting a number: token, identifier span, field or complete note. Placeholder precision and recall need alignment to the intended field or source span. Counting [NAME] tokens anywhere can credit a misplaced replacement. A direct leakage check scans for annotated source identifiers, using clear normalization and boundary rules. Clinical marker presence is a useful diagnostic, but preserving an unordered set of terms does not prove that relationships or negation survive. Score generated continuations, not the prompt echoed by the UI, and exclude only protocol control tokens according to a published rule. A zero-leakage result on a finite synthetic test is an observed rate, not proof of safety.

Sources: [metrics](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/metrics.py>), [data](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/data.py>), [assignment](<C:/Users/Nagar/source/repos/ai-learning-lab/topics/Mitra Topics/Week 6 — Build Your Own Tiny-GPT_Excercise.md>).

### 22. The demonstration should answer four questions

Planned demonstrations, with recorded outputs retained as a fallback

| Demonstration | What to show | Evidence required |
| --- | --- | --- |
| Complete generation | A fresh synthetic note, token by token | Actual output, EOS and preserved fields |
| Context matters | NSCLC / ASTHMA source pair | Changed probability at a matched output prefix |
| Temperature | Sampling at 0.1, 0.7 and 1.2 | Same checkpoint and decoding method |
| Failure and interpretation | One failure plus an inspected attention row | A counterfactual or ablation that tests the story |

Allocate around five minutes to the demo in the standard talk. First show a saved real output from the checkpoint, then run one live example. Explain the displayed source and expected target before generation. If the live model fails, interpret the failure rather than secretly switching to an authored target. Context tests should hold the output prefix and position constant. For attention inspection, capture the attention row corresponding to the prediction being explained, not simply the last model invocation before EOS. Label the selected layer, head, query token and key tokens. Treat an attention pattern as an interpretation and test it by changing an earlier source fact or ablating a head. Pre-record a short actual run for reliability.

Sources: [assignment](<C:/Users/Nagar/source/repos/ai-learning-lab/topics/Mitra Topics/Week 6 — Build Your Own Tiny-GPT_Excercise.md>), [generation](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/generation.py>), [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>).

### 23. Work remaining before the final presentation

Friday, October 23, 2026

| Stage | Concrete evidence to finish | Priority |
| --- | --- | --- |
| Data and correctness | Verified splits, masks, tokenizer coverage, tiny-batch learning | Required |
| A / B / C training | Saved checkpoints, config, seeds and validation selection | Required |
| Evaluation | Generated outputs, task metrics and held-out partitions | Required |
| Experiments | Context, temperature, failure and attention intervention | Required |
| Presentation finish | Measured result slides, recorded demo, timed rehearsal | Required |

This is the preparation-state closeout checklist. It should evolve into an evidence summary when the experiments are complete. Preserve the current honest status if the implementation is still unfinished. The required scope is a small educational comparison, not a new production system. Optional depth and tokenizer comparisons come after the required A/B/C runs. Freeze the evidence and presentation on Thursday rather than making risky model changes on Friday. The speaker must be able to trace a full prompt, explain every mask and distinguish teacher-forced scores from generated outputs. Completion of an artifact should not automatically mark learning mastery complete.

Sources: [assignment](<C:/Users/Nagar/source/repos/ai-learning-lab/topics/Mitra Topics/Week 6 — Build Your Own Tiny-GPT_Excercise.md>).

### 24. What this work teaches us

A GPT learns parameters that turn a prefix into the next-token distribution

Close the core talk with the distinction between mechanism and empirical capability. We can explain how a decoder works from code and arithmetic. We establish what it learned by observing controlled comparisons. The Week 5 evidence already teaches restraint: a larger architecture did not dominate every metric, and a very small test limits conclusions. Mitra adds the challenge of generating an exact transformation while preserving intended content. At the final presentation, answer the assignment question with actual examples of generalization, memorization and failure. Until experiments exist, say those questions remain open. An honest limited result with a clear explanation is stronger than a success story without evidence.

Sources: [assignment](<C:/Users/Nagar/source/repos/ai-learning-lab/topics/Mitra Topics/Week 6 — Build Your Own Tiny-GPT_Excercise.md>), [results](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/visualizations/arch_results.json>).

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

Sources: [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>), [attention](<https://arxiv.org/abs/1706.03762>).

### 26. Appendix: normalization and the FFN

Operations within a token representation

```text
LN(x) = γ ⊙ (x − μ) / √(σ² + ε) + β
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

LayerNorm uses features within one position, not statistics across all examples in a batch. With x=[1,3], mean is 2 and population variance is 1. Ignoring epsilon for this arithmetic and using gamma=[1,1], beta=[0,0], normalized output is [-1,1]. Learned gamma and beta can rescale and shift it. For the FFN, W1 has shape (64,192), b1 has shape (192), W2 has shape (192,64), and b2 has shape (64). The nonlinearity is essential: two linear maps with no nonlinear activation collapse into a single linear map. The starter uses GELU while the earlier NumPy study uses ReLU. Avoid claiming expansion itself guarantees richer measured behavior.

Sources: [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>), [numpy](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/numpy_transformer_suite.py>), [layernorm](<https://arxiv.org/abs/1607.06450>).

### 27. Appendix: the loss-mask boundary

The position holding <OUTPUT> predicts the first output token

| Input position token | Next target | Loss mask | Reason |
| --- | --- | --- | --- |
| <INPUT> | PATIENT | 0 | Part of source prediction |
| NSCLC | <OUTPUT> | 0 | Task boundary target |
| <OUTPUT> | PATIENT | 1 | First output token |
| PATIENT | [NAME] | 1 | Output placeholder |
| Final output NSCLC | <EOS> | 1 | Learn termination |
| <PAD> | <PAD> | 0 | Filler only |

This table is an excerpt, not one contiguous six-token input. The causal language-model input is the serialized sequence without its final token, and the target is that same sequence without its first token. At prediction index t, the target is original token t+1. If the index of <OUTPUT> in the original sequence is k, the target immediately after it is original index k+1 and must be scored. The starter expresses this using target_original_index > output_position. Check a hand-labeled example and ensure at least one scored target exists per training item. Do not silently truncate away the output continuation or EOS, and do not confuse a zero loss mask with a hidden attention key.

Sources: [data](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/data.py>), [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>).

### 28. Appendix: an illustrative parameter budget

Two blocks, d = 64, d_ff = 192, context = 96, untied embeddings

| Component | Count formula | If V = 128 |
| --- | --- | --- |
| Token embedding + output head | 2 × V × 64 | 16,384 |
| Position embedding | 96 × 64 | 6,144 |
| Attention projections, two blocks | 2 × 4 × 64² | 32,768 |
| FFNs with biases, two blocks | 2 × (2 × 64 × 192 + 192 + 64) | 49,664 |
| Block and final LayerNorms | 2 × 4 × 64 + 2 × 64 | 640 |
| Total | 128V + 89,216 | 105,600 |

This is a calculation for the inspected starter architecture, not the measured parameter count of a trained project. V=128 is a teaching assumption. Each attention block contains four bias-free 64 by 64 projections: Q, K, V and output. Each FFN has two weight matrices and both biases. Each block has two LayerNorms, with gamma and beta of width 64, and the full model adds one final LayerNorm. The vocabulary head is bias-free and does not tie its weights to the token embedding table. Actual V depends on the fitted training tokenizer, so use model.count_parameters for the final report. A fixed model width means changing the number of heads alone does not change these projection counts.

Sources: [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>), [config](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/config.py>).

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

Sources: [generation](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/generation.py>), [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>).

### 30. Appendix: starter issues to resolve

Code inspection identifies work, not experimental results

| Issue | Observed starter behavior | Required change or disclosure |
| --- | --- | --- |
| Template holdout | All four templates occur across splits | Add a genuinely held-out layout partition |
| ID isolation | MRN and phone use random sampling | Guarantee exact identifier disjointness |
| Metrics | Placeholder counts ignore field alignment | Add alignment and direct leakage checks |
| Decoding | Temperature threshold switches selection mode | Separate greedy and sampled generation |
| Context length | Dataset and generation truncate sequences | Reject or handle overflow with explicit reporting |

These are inspected properties of the local starter package. They are not claims that a trained model has failed or succeeded. Do not quietly claim a pure temperature experiment when the decoding method changes. Do not claim template generalization when the templates are shared. The placeholder metric can score a replacement in the wrong field because it intersects counts, so strengthen the metric before relying on it. Random IDs need an explicit disjointness check, and repeated synthetic examples need deduplication. The dataset slices to max_seq_len+1 and generation keeps only the most recent context window, which can remove required source or target information. The plan should resolve or clearly disclose these constraints.

Sources: [data](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/data.py>), [metrics](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/metrics.py>), [generation](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/generation.py>).

### 31. Appendix: questions for the technical discussion



Loss gradients reach source representations through attention dependencies even without direct source-target loss. Model A can learn local output grammar from the previous output token and absolute position, but cannot use a changed earlier source under a matched output prefix. Additional heads provide representational options, not guaranteed specialization or accuracy. A fixed vocabulary may exclude unseen names and therefore prevent literal emission; this requires known-name leakage tests and a coverage-aware tokenizer comparison. An unseen-template partition must hold out the structural pattern, not just names or random examples. On a constrained synthetic task, rules may be excellent. The learning contribution is understanding conditional generation and empirically testing its limits, not declaring a production advantage.

Sources: [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>), [tokenizer](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/tokenizer.py>), [data](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/data.py>), [results](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/visualizations/arch_results.json>).

### 32. Appendix: evidence and references



Primary architecture source: https://arxiv.org/abs/1706.03762 . Normalization: https://arxiv.org/abs/1607.06450 . Interpretation boundaries: https://aclanthology.org/N19-1357/ and https://aclanthology.org/D19-1002/ . Local reference selection followed resources/references/REFERENCE_MAP.md. Targeted tokenizer intuition came from Hands-On Large Language Models, Chapter 2. Learning-objective intuition came from Deep Learning, Chapter 5 maximum likelihood discussion. The local file labeled Build a Large Language Model (From Scratch) is excluded because the library inventory identifies mismatched contents. Numerical project results come from the saved Week 5 JSON. All toy attention values and parameter budgets are clearly labeled illustrative.

Sources: [assignment](<C:/Users/Nagar/source/repos/ai-learning-lab/topics/Mitra Topics/Week 6 — Build Your Own Tiny-GPT_Excercise.md>), [results](<C:/Users/Nagar/source/repos/ai-learning-lab/projects/week 5/visualizations/arch_results.json>), [model](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/model.py>), [data](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/data.py>), [metrics](<C:/Users/Nagar/source/repos/tiny-mitra-week6-project-v2/tiny-mitra/tiny_mitra/metrics.py>), [attention](<https://arxiv.org/abs/1706.03762>), [layernorm](<https://arxiv.org/abs/1607.06450>), [interpretation](<https://aclanthology.org/N19-1357/>), [interpretation2](<https://aclanthology.org/D19-1002/>).
