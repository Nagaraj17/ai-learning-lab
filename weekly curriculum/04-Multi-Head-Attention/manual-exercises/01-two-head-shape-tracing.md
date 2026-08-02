# Manual Exercise 1: Two-Head Shape Tracing

> Objective: trace every major tensor shape in a tiny two-head attention layer
> before relying on code.

## Given

- sequence length `T = 4`
- model dimension `d_model = 4`
- number of heads `h = 2`
- per-head dimensions `d_k = d_v = 2`

### Input

`X` has shape `(4, 4)`.

Each head uses its own:

- `W_Q^i` with shape `(4, 2)`
- `W_K^i` with shape `(4, 2)`
- `W_V^i` with shape `(4, 2)`

## Learner Workspace

### Step 1

What is the shape of:

- `Q_1 = X W_Q^1`
- `K_1 = X W_K^1`
- `V_1 = X W_V^1`

Answer:

- `Q_1`:
- `K_1`:
- `V_1`:

### Step 2

What is the shape of `K_1^T`?

Answer:

### Step 3

What is the shape of:

- `scores_1 = Q_1 K_1^T`
- `attn_1 = softmax(scores_1 / sqrt(d_k))`
- `head_1 = attn_1 V_1`

Answer:

- `scores_1`:
- `attn_1`:
- `head_1`:

### Step 4

Repeat the same shape reasoning for head 2.

Answer:

- `Q_2`:
- `K_2`:
- `V_2`:
- `scores_2`:
- `attn_2`:
- `head_2`:

### Step 5

If `head_1` and `head_2` are both `(4, 2)`, what is the shape of:

- `concat = concatenate([head_1, head_2], axis=1)`

Answer:

### Step 6

If the final output must return to model width `d_model = 4`, what shape should
`W_O` have?

Answer:

### Step 7

What is the final output shape of:

`output = concat W_O`

Answer:

## Solution Key

- `Q_1`, `K_1`, `V_1`: `(4, 2)`
- `K_1^T`: `(2, 4)`
- `scores_1`: `(4, 4)`
- `attn_1`: `(4, 4)`
- `head_1`: `(4, 2)`
- `Q_2`, `K_2`, `V_2`: `(4, 2)`
- `scores_2`: `(4, 4)`
- `attn_2`: `(4, 4)`
- `head_2`: `(4, 2)`
- `concat`: `(4, 4)`
- `W_O`: `(4, 4)`
- `output`: `(4, 4)`

## Why This Exercise Matters

If you can trace these shapes confidently, the code becomes much less
mysterious. Most Multi-Head Attention bugs at this stage are shape bugs.
