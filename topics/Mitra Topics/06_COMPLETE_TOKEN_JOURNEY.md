# Topic 6 — Complete Token Journey: Input to Next-Token Prediction

## Learning goal

This is the end-to-end walkthrough every team member should be able to give. It connects the earlier topics using one prompt and concrete tensor shapes.

## The prompt

```text
<BOS> <INPUT> PATIENT Olivia Martinez DIAGNOSIS NSCLC <OUTPUT>
```

Desired continuation:

```text
PATIENT [NAME] DIAGNOSIS NSCLC <EOS>
```

Assume:

| Setting | Value |
|---|---:|
| Batch `B` | 1 |
| Prompt length `T` | 8 |
| Model width `D` | 24 |
| Heads `H` | 4 |
| Head width `d_head` | 6 |
| FFN width `d_ff` | 96 |
| Blocks | 2 |
| Vocabulary `V` | 80 |

The exact vocabulary IDs and learned values vary after training. Shapes and operations do not.

## Stage 1 — Tokenization

The regex produces:

```text
[
  "<BOS>", "<INPUT>", "PATIENT", "Olivia",
  "Martinez", "DIAGNOSIS", "NSCLC", "<OUTPUT>"
]
```

The sequence length is `T=8`.

If `Olivia` and `Martinez` are absent from the training vocabulary:

```text
[
  <BOS_ID>, <INPUT_ID>, <PATIENT_ID>, <UNK_ID>,
  <UNK_ID>, <DIAGNOSIS_ID>, <NSCLC_ID>, <OUTPUT_ID>
]
```

Tensor:

```text
input_ids: (B,T) = (1,8)
```

The two unknown IDs are numerically identical, but their positions differ.

## Stage 2 — Token embedding lookup

The embedding table has:

```text
(V,D) = (80,24)
```

Each token ID selects one row:

```text
(1,8) → embedding lookup → (1,8,24)
```

No multiplication by the integer ID occurs. An ID is an index into a learned table.

Both `<UNK>` positions initially retrieve the same token embedding. Their representations become different after positional vectors are added and context is processed.

## Stage 3 — Positional embedding

Position IDs are:

```text
[0,1,2,3,4,5,6,7]
```

The position table has shape:

```text
(max_seq_len,D)
```

Selected position vectors have shape:

```text
(T,D) = (8,24)
```

Broadcasting adds them to every batch:

```python
x = token_embedding(input_ids) + position_embedding(positions)[None,:,:]
```

Result:

```text
x: (1,8,24)
```

Now the two `<UNK>` tokens differ because one is at position 3 and one at position 4.

## Stage 4 — Enter Decoder Block 1

The block input is `x₀: (1,8,24)`.

Pre-LayerNorm normalizes each 24-feature token vector independently:

```text
n₀ = LayerNorm(x₀): (1,8,24)
```

It changes values, not shape.

## Stage 5 — Q, K, and V

Three independent linear projections:

```text
Q = n₀ @ W_Q
K = n₀ @ W_K
V = n₀ @ W_V
```

Each weight is `(24,24)`, so each result is:

```text
Q, K, V: (1,8,24)
```

The `<OUTPUT>` position has its own query vector. All eight positions have key and value vectors.

## Stage 6 — Split into four heads

```text
(1,8,24)
→ reshape (1,8,4,6)
→ transpose (1,4,8,6)
```

So:

```text
Q, K, V: (1,4,8,6)
```

Each head has six projected features per position. The four heads together still represent 24 features.

## Stage 7 — Score every visible source position

```text
scores = Q @ Kᵀ / √6
```

Shape:

```text
(1,4,8,6) @ (1,4,6,8) → (1,4,8,8)
```

For the `<OUTPUT>` query at index 7:

```text
scores[0, head, 7, 0:8]
```

contains one compatibility score for every prompt position:

```text
<BOS>, <INPUT>, PATIENT, <UNK>, <UNK>, DIAGNOSIS, NSCLC, <OUTPUT>
```

## Stage 8 — Apply causal and padding masks

At position 7, all prompt positions are at or before it, so all are causally visible.

At position 3, only positions 0 through 3 are visible; positions 4 through 7 receive `-inf`.

There is no padding in this eight-token prompt, so the padding mask is all ones. If padding existed, its key columns would also receive `-inf`.

This is the exact point where future information is prevented from leaking.

## Stage 9 — Softmax attention weights

Softmax across the last dimension turns each row into probabilities:

```text
weights: (1,4,8,8)
```

Every valid row per head sums to one.

An illustrative `<OUTPUT>` row for one head could be:

```text
[0.02, 0.03, 0.30, 0.20, 0.20, 0.10, 0.10, 0.05]
```

This head pays most attention to `PATIENT` and the two unknown positions. These are illustrative values, not guaranteed learned behaviour.

## Stage 10 — Weighted value combination

```text
context = weights @ V
```

```text
(1,4,8,8) @ (1,4,8,6) → (1,4,8,6)
```

For each query and head, source value vectors are mixed using the attention probabilities.

## Stage 11 — Merge heads and project

```text
(1,4,8,6)
→ transpose (1,8,4,6)
→ reshape (1,8,24)
→ output projection (1,8,24)
```

The output projection learns how features from different heads should interact.

## Stage 12 — First residual update

```text
x₁ = x₀ + attention_output
```

Both tensors are `(1,8,24)`. The result is also `(1,8,24)`.

The original representation follows the residual path; attention contributes a learned contextual update.

## Stage 13 — Feed-forward update

Normalize:

```text
n₁ = LayerNorm(x₁): (1,8,24)
```

Expand:

```text
(1,8,24) @ (24,96) → (1,8,96)
```

Apply GELU, then contract:

```text
(1,8,96) @ (96,24) → (1,8,24)
```

Second residual:

```text
x₂ = x₁ + FFN(n₁): (1,8,24)
```

The FFN transforms every position independently. The position-to-position mixing already occurred in attention.

## Stage 14 — Decoder Block 2

Block 2 receives `(1,8,24)` and repeats the same sequence with its own learned weights:

```text
LayerNorm → Q/K/V → four-head causal attention → residual
→ LayerNorm → FFN → residual
```

It again outputs `(1,8,24)`.

Shape stability does not mean nothing changed. The 24 features at each position now encode richer contextual information.

## Stage 15 — Final normalization

After all blocks:

```text
hidden = final_norm(x): (1,8,24)
```

The model has one final 24-feature hidden representation for every input position.

## Stage 16 — Vocabulary projection

The learned output matrix has shape:

```text
(D,V) = (24,80)
```

```text
(1,8,24) @ (24,80) → logits (1,8,80)
```

Every position now has 80 scores—one for each vocabulary token.

## Stage 17 — Select final-position logits

```text
logits[:, -1, :] → (1,80)
```

These scores answer:

> Given the entire prompt ending in `<OUTPUT>`, what token should come next?

Earlier-position logits are useful during parallel training, but only the final current position is needed for generation.

## Stage 18 — Temperature and probability

At temperature `0.1`:

```text
adjusted = final_logits / 0.1
probabilities = softmax(adjusted)
```

Suppose `PATIENT` is highest. Greedy selection chooses its ID.

## Stage 19 — Append and repeat

New sequence:

```text
<BOS> <INPUT> PATIENT Olivia Martinez DIAGNOSIS NSCLC <OUTPUT> PATIENT
```

Now `T=9`. The model repeats token embedding, positional embedding, every decoder block, final normalization, and vocabulary projection.

The new final-position distribution should ideally choose `[NAME]`.

This continues:

```text
PATIENT
PATIENT [NAME]
PATIENT [NAME] DIAGNOSIS
PATIENT [NAME] DIAGNOSIS NSCLC
PATIENT [NAME] DIAGNOSIS NSCLC <EOS>
```

Generation stops at `<EOS>`.

## Complete shape table for the first token

| Stage | Shape |
|---|---|
| Token IDs | `(1,8)` |
| Token embeddings | `(1,8,24)` |
| Position embeddings | `(8,24)` |
| Combined input | `(1,8,24)` |
| Q/K/V before heads | each `(1,8,24)` |
| Q/K/V after heads | each `(1,4,8,6)` |
| Attention scores | `(1,4,8,8)` |
| Attention weights | `(1,4,8,8)` |
| Head contexts | `(1,4,8,6)` |
| Merged attention | `(1,8,24)` |
| Block output | `(1,8,24)` |
| Final hidden states | `(1,8,24)` |
| All-position logits | `(1,8,80)` |
| Final-position logits | `(1,80)` |
| Selected next token | one integer ID |

## What training did differently

During training, the complete expected output was already included:

```text
... <OUTPUT> PATIENT [NAME] DIAGNOSIS NSCLC <EOS>
```

Input and targets were shifted by one position. Causal masking prevented each position from inspecting its target or anything after it. Loss was applied only to targets after `<OUTPUT>`.

Thus training predicted all supervised output positions in parallel without future leakage. Inference must construct those output positions one at a time.

## Code map

| Concept | Implementation |
|---|---|
| Split strings | `WordTokenizer.split` |
| IDs | `WordTokenizer.encode` |
| Shift and loss mask | `CausalLMDataset.__getitem__` |
| Token/position vectors | `TinyLanguageModel.forward` |
| Q/K/V and heads | `ManualMultiHeadCausalAttention` |
| Causal/padding masks | `ManualMultiHeadCausalAttention.forward` |
| Residual + LayerNorm + FFN | `DecoderBlock.forward` |
| Vocabulary logits | `vocabulary_projection` |
| Output-only cross-entropy | `masked_cross_entropy` |
| Predict/append loop | `generate` |

## Questions the instructor may interrupt with

1. Why are both unknown name tokens initially represented by the same embedding?
2. What makes them different before attention?
3. Which dimensions are multiplied to create `T×T` scores?
4. Where exactly is the future set to `-inf`?
5. Along which dimension is softmax applied?
6. What do attention weights multiply?
7. Why does concatenating four six-dimensional heads return 24?
8. Does the FFN mix sequence positions?
9. Why can residual addition occur?
10. Why do we read the last position during generation?
11. Why is the complete model rerun after appending one token?
12. Why is this de-identification still next-token prediction?

## One-minute explanation

The prompt is split into tokens and converted to IDs. Each ID retrieves a token embedding, and a positional embedding is added so order is represented. Inside every decoder block, LayerNorm prepares the vectors, separate linear layers create Q, K, and V, and the features are divided into heads. Q and K create scaled `T×T` compatibility scores. The causal mask sets future scores to negative infinity, softmax creates attention weights, and those weights mix the value vectors. Head outputs are concatenated, projected, and added through a residual connection. A normalized feed-forward network expands and contracts each token representation, followed by another residual. After all blocks, the final hidden vectors are projected to vocabulary logits. We take logits at the last prompt position, apply temperature and softmax, select one token, append it, and repeat until `<EOS>`. Therefore the redacted note is generated one next-token prediction at a time.

## Key takeaway

At every generation step, one sequence becomes `(B,T,D)` contextual representations, then `(B,T,V)` vocabulary logits, and finally one selected next token. Appending that token starts the same journey again.
