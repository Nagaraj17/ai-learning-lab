# Topic 3 — Building the GPT Causal Decoder Block

## Learning goal

You should be able to take a tensor of shape `(B, T, D)`, trace it through every operation in a decoder block, explain the purpose of each component, and show why the shape returns to `(B, T, D)`.

## 1. Transformer block versus decoder block

**Transformer block** is a family term. Encoder blocks, decoder blocks, and architecture-specific variations are all Transformer blocks.

GPT uses a **decoder-only** architecture. Its self-attention is causal: a position may read itself and earlier positions but not later positions.

Our block is Pre-LayerNorm:

```text
x
├─ LayerNorm → causal multi-head attention ─┐
└──────────────── residual addition ────────┘ → x₁

x₁
├─ LayerNorm → feed-forward network ────────┐
└──────────────── residual addition ─────────┘ → x₂
```

In code:

```python
x = x + self.attention(self.attention_norm(x), padding_mask)
x = x + self.ffn(self.ffn_norm(x))
```

## 2. Symbols and a concrete example

| Symbol | Meaning | Example |
|---|---|---:|
| `B` | Batch size | 2 |
| `T` | Sequence length | 10 |
| `D` | Model width, `d_model` | 24 |
| `H` | Number of heads | 4 |
| `d_head` | Width per head, `D/H` | 6 |
| `d_ff` | FFN hidden width | 96 |
| `V` | Vocabulary size | 80 |

The block input is:

```text
x: (2, 10, 24)
```

There are 2 examples, 10 positions per example, and a 24-number representation per token.

## 3. LayerNorm

LayerNorm normalizes each token’s feature vector independently. For one token vector with `D` values, it computes its mean and variance, normalizes, then applies learned scale and shift.

```text
LN(x) = learned_scale × normalized(x) + learned_shift
```

It does not mix examples or sequence positions. Shape remains:

```text
(B, T, D) → (B, T, D)
```

Pre-LayerNorm means normalization occurs before attention and before the FFN. This generally makes deeper residual networks easier to optimize because the residual route remains direct.

## 4. Q, K, and V projections

The normalized representation is passed through three independently learned linear layers:

```python
Q = x @ W_Q
K = x @ W_K
V = x @ W_V
```

Each weight matrix is `(D, D)`, so:

```text
(B, T, D) @ (D, D) → (B, T, D)
```

For `D = 24`, every matrix has shape `24 × 24`.

### Intuition

- **Query:** what information is this position looking for?
- **Key:** what kind of information does this source position offer?
- **Value:** what content should be transferred if the query and key match?

Q, K, and V begin from the same input tensor but use different learned weights, so they play different roles.

## 5. Splitting into heads

Our code reshapes:

```text
(B, T, D)
→ (B, T, H, d_head)
→ transpose
→ (B, H, T, d_head)
```

With `D=24`, `H=4`, and `d_head=6`:

```text
(2, 10, 24) → (2, 4, 10, 6)
```

The 24 projected features are divided among four heads. The total projection width remains 24; it is not 24 features per head.

Increasing heads while keeping `D` fixed changes how features are partitioned, not the total Q/K/V parameter count:

```text
W_Q + W_K + W_V + W_O = 4 × D × D
```

ignoring biases. For `D=24`, that is `4 × 24 × 24 = 2,304` weights.

## 6. Attention scores

For every head:

```text
scores = Q @ Kᵀ / √d_head
```

Shapes:

```text
Q:  (B, H, T, d_head)
Kᵀ: (B, H, d_head, T)
scores: (B, H, T, T)
```

The first `T` indexes the querying position. The second `T` indexes the key/source position.

One score cell:

```text
scores[b, h, i, j]
```

means: in batch example `b` and head `h`, how compatible is the query at position `i` with the key at position `j`?

## 7. Why divide by `√d_head`

Dot products tend to grow in magnitude when they sum across more dimensions. Large logits push softmax toward extremely sharp probabilities, where gradients may become weak.

For `d_head=6`:

```text
scale = √6 ≈ 2.45
```

A raw score of `4.9` becomes about `2.0`. Scaling does not change which score is largest; it controls magnitude before softmax.

## 8. Causal masking

For `T=5`, allowed positions look like:

```text
Query 0: ✓ ✗ ✗ ✗ ✗
Query 1: ✓ ✓ ✗ ✗ ✗
Query 2: ✓ ✓ ✓ ✗ ✗
Query 3: ✓ ✓ ✓ ✓ ✗
Query 4: ✓ ✓ ✓ ✓ ✓
```

The code creates an upper triangular Boolean mask above the diagonal:

```python
causal_mask = torch.triu(
    torch.ones(time, time, dtype=torch.bool),
    diagonal=1
)
scores = scores.masked_fill(causal_mask, float("-inf"))
```

After softmax, an `-inf` score receives probability zero.

### Why this prevents cheating

When the representation at position `t` predicts token `t+1`, it must not inspect that future token’s input representation. Otherwise training loss becomes artificially excellent, but generation fails because future output tokens are unavailable.

### Can the final prompt position read the entire prompt?

Yes. The `<OUTPUT>` position is the latest prompt position, so all prompt tokens are on its left. It may attend to them all when predicting the first redacted token.

## 9. Padding masking

Causal masking hides future positions. Padding masking hides artificial `<PAD>` keys:

```python
key_is_padding = padding_mask[:, None, None, :] == 0
scores = scores.masked_fill(key_is_padding, float("-inf"))
```

Shape broadcasting changes `(B, T)` to `(B, 1, 1, T)`, applying the same invalid-key positions to every head and every query.

## 10. Softmax and weighted values

Softmax is applied along the final dimension—the key positions:

```python
weights = torch.softmax(scores, dim=-1)
```

Every query row for every head sums to 1. For one query:

```text
weights = [0.10, 0.60, 0.30, 0, 0]
```

The context is:

```text
0.10×V₀ + 0.60×V₁ + 0.30×V₂
```

Code:

```python
context = weights @ value
```

Shapes:

```text
(B,H,T,T) @ (B,H,T,d_head) → (B,H,T,d_head)
```

Attention weights say **where to read**. Value vectors contain **what is read**.

## 11. Merge heads and output projection

The contexts are transposed and reshaped:

```text
(B,H,T,d_head)
→ (B,T,H,d_head)
→ (B,T,D)
```

Then:

```python
output = merged @ W_O
```

`W_O` has shape `(D,D)`. It learns how to mix the head features back into one model-width representation.

## 12. First residual connection

```text
x₁ = x + Attention(LayerNorm(x))
```

Both terms must have `(B,T,D)`, which is why attention returns to width `D`.

The residual does not mean attention is ignored. It gives the next layer access to both the old representation and the learned attention update. It also provides a shorter gradient path during backpropagation.

## 13. Feed-forward network

The FFN is applied independently to each token position:

```text
(B,T,D)
→ Linear(D,d_ff)
→ GELU
→ Linear(d_ff,D)
→ (B,T,D)
```

For `D=24`, `d_ff=96`:

```text
(2,10,24) → (2,10,96) → (2,10,24)
```

Attention mixes information **between positions**. The FFN transforms features **within each position** using the same learned network at every position.

The expansion provides a larger workspace for nonlinear feature combinations. Contracting back to `D` makes the residual addition and block stacking possible.

## 14. Second residual and stacked blocks

```text
x₂ = x₁ + FFN(LayerNorm(x₁))
```

One decoder block outputs `(B,T,D)`, so another block can accept it directly. Although shape remains unchanged, values and meaning evolve. Later blocks can build on contextual features produced by earlier blocks.

## 15. Final normalization and vocabulary projection

After the block stack:

```python
logits = vocabulary_projection(final_norm(x))
```

```text
(B,T,D) @ (D,V) → (B,T,V)
```

For every position, the model now has one logit per vocabulary token. Generation reads only the final position:

```text
logits[:, -1, :] → (B,V)
```

## 16. Exact shape journey

Using `B=2, T=10, D=24, H=4, d_head=6, d_ff=96, V=80`:

| Stage | Shape |
|---|---|
| Block input | `(2,10,24)` |
| Normalized input | `(2,10,24)` |
| Q, K, V before split | each `(2,10,24)` |
| Q, K, V after split | each `(2,4,10,6)` |
| Scores | `(2,4,10,10)` |
| Attention weights | `(2,4,10,10)` |
| Per-head context | `(2,4,10,6)` |
| Merged context | `(2,10,24)` |
| Attention output | `(2,10,24)` |
| First residual result | `(2,10,24)` |
| FFN expanded | `(2,10,96)` |
| FFN contracted | `(2,10,24)` |
| Block output | `(2,10,24)` |
| Vocabulary logits | `(2,10,80)` |

## 17. Common misconceptions

**“Every head gets all `D` features.”**  
Not here. Each head gets `D/H` projected features.

**“More heads automatically means more parameters.”**  
Not when `D` stays fixed and combined Q/K/V widths remain `D`.

**“The FFN attends to other tokens.”**  
No. Attention mixes positions; FFN operates position-wise.

**“Causal masking deletes future tokens.”**  
No. It sets their attention scores to `-inf` for earlier queries.

**“Residual connections preserve the original input unchanged forever.”**  
They add a bypass path, but every block still accumulates learned updates.

## 18. Checks and experiments

1. Assert all attention weights above the diagonal are zero.
2. Assert every valid attention row sums to one.
3. Print shapes inside `_split_heads`.
4. Change one future token and verify earlier-position logits remain unchanged in evaluation mode.
5. Compare one head versus four heads with the same `D`.
6. Remove the `√d_head` scaling and observe softmax sharpness.
7. Temporarily remove a residual connection and compare training stability.

## 19. Explain without notes

Take the `<OUTPUT>` position and explain:

1. how its vector becomes Q, K, and V;
2. why its query can read every prompt token;
3. why a much earlier token cannot read `<OUTPUT>`;
4. why scores have two `T` dimensions;
5. what softmax weights and values do differently;
6. why heads must be merged;
7. why FFN expands and contracts;
8. how the final `D` values become `V` logits.

## Key takeaway

A decoder block repeatedly performs two learned updates: causal attention gathers relevant earlier context, and the FFN transforms the resulting features. LayerNorm stabilizes the inputs, residual paths preserve and update information, and fixed `(B,T,D)` dimensions allow blocks to stack.
