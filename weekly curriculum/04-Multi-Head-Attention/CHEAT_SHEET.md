# Week 4 Cheat Sheet

## Core Idea

One head gives one attention pattern and one contextual output stream.
Multi-Head Attention runs several heads in parallel on the **same input
sequence** using different learned projections.

## Minimal Vocabulary

- `T`: sequence length
- `d_model`: model width
- `h`: number of heads
- `d_k`: per-head query/key width
- `d_v`: per-head value width
- `W_Q^i`, `W_K^i`, `W_V^i`: head `i` projection matrices
- `W_O`: output projection after concatenation

## Key Corrections

- Every head processes the full sequence.
- Heads do not manually receive business roles.
- Specialization may emerge, but it is not guaranteed.
- Some heads may be redundant.
- Multi-Head Attention is not Mixture-of-Experts.
- Multi-Head Attention does not remove `O(T^2)` sequence-length cost.

## Formula

For each head:

`head_i = softmax((Q_i K_i^T) / sqrt(d_k)) V_i`

where:

- `Q_i = X W_Q^i`
- `K_i = X W_K^i`
- `V_i = X W_V^i`

Full layer:

`MultiHead(X) = Concat(head_1, ..., head_h) W_O`

## Common Shape Pattern

If:

- `X`: `(T, d_model)`
- `h = 2`
- `d_k = d_v = d_model / h`

then each head has:

- `Q_i`, `K_i`, `V_i`: `(T, d_k)`
- scores: `(T, T)`
- attention weights: `(T, T)`
- head output: `(T, d_v)`

After concatenation:

- `(T, h * d_v)`

After `W_O`:

- `(T, d_model)`

## Why `d_k = d_model / h` Is Common

- it keeps the concatenated width equal to `d_model`
- it keeps total attention computation comparable to one full-width head

It is a standard design choice, not the universal mathematical definition.

## Why `W_O` Exists

Concatenation only stacks head outputs.
`W_O` learns how to mix those outputs and return to model space.

## What to Say Carefully

Safe:

- "Different heads can learn different useful patterns."
- "The experiment suggests head 2 became more important for this toy task."
- "Some heads may be pruned with limited loss increase."

Unsafe without evidence:

- "Head 1 learned contracts."
- "Head 2 learned finance."
- "More heads always help."

## Debug Checklist

- Did every head receive the same `X`?
- Are the per-head projection matrices independent?
- Do score matrices have shape `(T, T)`?
- Do attention rows sum to approximately `1.0`?
- Is causal masking applied inside each head when needed?
- Does concatenation produce the expected width?
- Does `W_O` map the concatenated width back correctly?

## Quick Comparison

Single-head:

- one set of `Q/K/V`
- one attention matrix
- one output stream

Multi-head:

- `h` sets of `Q/K/V`
- `h` attention matrices
- `h` output streams
- concatenate
- apply `W_O`
