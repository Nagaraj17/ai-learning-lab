# 25 - TRANSFORMER - Causal Masking

## 1. The Problem
In autoregressive language models (like GPT-4), the task is to **predict the next token** $t_{i+1}$ given previous tokens $[t_1, \dots, t_i]$.
If we use standard un-masked Self-Attention, token $i$ can look forward into position $i+1$ and read the exact answer it is supposed to predict.
**The limitation:** Un-masked self-attention allows "cheating" during training; it allows the training computation to access information that should not be available under the autoregressive factorization.

## 2. Why We Need Something New
We need a mathematical filter (**Causal Masking**) that blocks attention scores from position $i$ to any future position $j > i$, ensuring token $i$ can ONLY attend to past and present tokens ($j \le i$).

## 3. One-Line Definition
**Causal Masking** is the technique of adding an upper-triangular matrix of $-\infty$ values to the raw attention score matrix $\mathbf{S}$ before Softmax, driving future token attention probabilities to exactly $0.0$.

## 4. Beginner Intuition / Mental Model
Imagine taking an exam where the answers to future questions are printed on the back page. 
**Causal Masking** is like placing a **cardboard barrier** over all future pages: you can only look at questions you have already answered, preventing you from peeking ahead.

## 5. What Came Before → What Changes Now
- **BERT-style encoder self-attention:** Bidirectional / unmasked with respect to future token positions. Every token attends to all $T$ tokens (upper and lower triangle active).
- **Transformer decoder self-attention:** Causally masked so positions cannot attend to subsequent output positions.
- **GPT-style decoder-only Transformer:** Causally masked self-attention. Token $i$ attends ONLY to tokens $1 \dots i$ (upper triangle set to 0% probability).

## 6. How It Works
1. Create a lower-triangular boolean mask matrix $\mathbf{M}_{\text{bool}} \in \mathbb{R}^{T \times T}$ where valid past positions ($j \le i$) are `True` and future positions ($j > i$) are `False`.
2. Convert `False` entries to $-\infty$ (or a large negative number like $-1e9$) and `True` entries to $0.0$ to construct mask matrix $\mathbf{M} \in \mathbb{R}^{T \times T}$.
3. Add mask matrix $\mathbf{M}$ to scaled dot-product scores: $\mathbf{S}_{\text{masked}} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}$.
4. Apply Softmax: $e^{-\infty} = 0.0$, guaranteeing $0\%$ attention to future tokens!

```
Raw Scores S         +     Causal Mask M      =    Masked Scores       ──►  Softmax Attention A
[ s11  s12  s13 ]        [   0   -inf  -inf ]      [  s11  -inf  -inf ]       [ 1.00  0.00  0.00 ]  (Token 1 attends only to Token 1)
[ s21  s22  s23 ]        [   0     0   -inf ]      [  s21   s22  -inf ]       [ a21   a22   0.00 ]  (Token 2 attends to Tokens 1, 2)
[ s31  s32  s33 ]        [   0     0     0  ]      [  s31   s32   s33 ]       [ a31   a32   a33  ]  (Token 3 attends to Tokens 1, 2, 3)
```

## 7. Required Mathematics
For sequence length $T$:

$$M_{i, j} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$

$$\mathbf{A} = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M} \right) \in \mathbb{R}^{T \times T}$$

**Shape Trace:**
- $\mathbf{Q} \mathbf{K}^\top$: $(T \times T)$
- Mask $\mathbf{M}$: $(T \times T)$
- $\mathbf{S}_{\text{masked}}$: $(T \times T)$
- Attention weights $\mathbf{A}$: $(T \times T)$ (Upper triangle entries are exactly $0.0$).

### Symbol Table

| Symbol | Name | Plain-English Meaning |
| :--- | :--- | :--- |
| $\mathbf{M}$ | **Causal Mask Matrix** | A $(T \times T)$ matrix of $0$s and $-\infty$s. $0$ in positions where attention is allowed ($j \le i$), and $-\infty$ where it is blocked ($j > i$). |
| $-\infty$ | **Negative Infinity** | A very large negative number (practically $-10^9$ in code). When exponentiated by Softmax, $e^{-\infty} = 0.0$, guaranteeing zero attention weight. |
| $M_{i,j}$ | **Mask Entry** | The scalar at row $i$, column $j$ of the mask. $M_{i,j} = 0$ means "token $i$ is allowed to see token $j$". $M_{i,j} = -\infty$ means "token $i$ is blocked from seeing token $j$". |
| $\mathbf{S}_{\text{masked}}$ | **Masked Score Matrix** | The result of $\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}$. Future positions now contain $-\infty$. |
| $j \le i$ | **Past/Present Condition** | Token at position $i$ can attend to any token at position $j$ that is at or before itself. |
| $j > i$ | **Future Condition** | Token at position $i$ is BLOCKED from attending to any token at position $j$ that is after itself. |
| `np.tril()` | **Lower-Triangular Function** | NumPy function that creates a matrix with $1$s on and below the diagonal and $0$s above. Used to build the causal mask. |

## 8. Complete Worked Example
Let $T = 3$. Raw scaled scores $\mathbf{S}_{\text{scaled}} = \begin{bmatrix} 2.0 & 5.0 & 1.0 \\ 3.0 & 4.0 & 6.0 \\ 1.0 & 2.0 & 3.0 \end{bmatrix} \in \mathbb{R}^{3 \times 3}$.

Notice row 0 (`"Order"`): raw score to `"Shipment"` is $5.0$. Without a mask, Token 0 would peek at Token 1!

Add Causal Mask $\mathbf{M} = \begin{bmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ 0 & 0 & 0 \end{bmatrix}$:

$$\mathbf{S}_{\text{masked}} = \begin{bmatrix} 2.0 & -\infty & -\infty \\ 3.0 & 4.0 & -\infty \\ 1.0 & 2.0 & 3.0 \end{bmatrix}$$

Apply row-wise Softmax:
- Row 0: $\text{Softmax}([2.0, -\infty, -\infty]) = [1.0, 0.0, 0.0]$
- Row 1: $\text{Softmax}([3.0, 4.0, -\infty]) \approx [0.269, 0.731, 0.0]$
- Row 2: $\text{Softmax}([1.0, 2.0, 3.0]) \approx [0.090, 0.245, 0.665]$

$$\mathbf{A} = \begin{bmatrix} 1.000 & 0.000 & 0.000 \\ 0.269 & 0.731 & 0.000 \\ 0.090 & 0.245 & 0.665 \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$

Future positions receive **exactly 0% weight**!

## 9. Math → Code Mapping
```python
import numpy as np

T = 3
scores = np.array([
    [2.0, 5.0, 1.0],
    [3.0, 4.0, 6.0],
    [1.0, 2.0, 3.0]
])

# Create lower-triangular causal mask
mask = np.tril(np.ones((T, T))) # 1 for valid, 0 for future
mask_scores = np.where(mask == 1, scores, -1e9)

# Softmax
exp_s = np.exp(mask_scores - np.max(mask_scores, axis=-1, keepdims=True))
A = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

print("Causal Attention Weights A:\n", np.round(A, 3))
```

## 10. Experiments / What-If Questions
- **What happens if we set masked values to $0.0$ instead of $-\infty$ before Softmax?**
  Softmax computes $e^0 = 1.0$. Setting masked entries to $0.0$ gives future positions positive non-zero attention probabilities ($\frac{1}{\sum e^z}$)! They MUST be set to $-\infty$ because $e^{-\infty} = 0.0$.

## 11. Common Misunderstandings
- **Misunderstanding:** Causal masking is used in all Transformers, including BERT.
- **Correction:** Causal masking is used wherever autoregressive self-attention must prevent access to future positions (e.g., the original Transformer decoder and decoder-only models like GPT). Encoder models (BERT) use bidirectional (unmasked) attention.
- **Misunderstanding:** Code implementations literally use $-\infty$.
- **Correction:** While $-\infty$ is the clean mathematical value, code uses large finite negative values (e.g., `-1e9`, or framework masking APIs) as practical approximations.

## 12. Limitations and Trade-Offs
Causal masking restricts token 1 to only see itself, limiting early tokens from gathering future context. However, it is mandatory for autoregressive generation.

## 13. Where It Appears in the Current Assignment
In **Week 3 Assignment**, you will implement causal masking using `np.tril()` and verify that the upper triangle of attention weights `A` is strictly $0.0$.

## 14. Where It Appears in Modern AI Systems
Causal masking is a standard component of autoregressive Transformer decoders.

## 15. Connection to the Next Concept
How do we allow the model to focus on multiple different context relationships simultaneously? That requires **Multi-Head Attention** (`26 - TRANSFORMER - Multi-Head Attention.md`).

## 16. Teach-Back and Small Application Exercise
1. What value must be added to future position scores before Softmax, and why?
2. In a sequence of length $T = 4$, how many entries in attention matrix $\mathbf{A}$ will be zero due to causal masking?

> **Hint for Q2:** The zero entries form the **strictly upper triangle** of a $T \times T$ matrix. The number of entries in a strict upper triangle is $\frac{T(T-1)}{2}$. For $T = 4$: $\frac{4 \times 3}{2} = 6$ entries are masked to zero.

## 17. Quick Revision Summary
- Causal Masking prevents token $i$ from attending to future tokens $j > i$.
- Adds $-\infty$ to upper-triangular scores before Softmax ($e^{-\infty} = 0.0$).
- Essential for autoregressive next-token prediction.

## 18. My Understanding
*Fill in your own summary of how Causal Masking enforces autoregressive bounds.*

## 19. Flashcards
Why must causal masking use $-\infty$ instead of $0$ in the score matrix before Softmax? #card
Because $\text{Softmax}$ exponentiates its inputs ($e^x$). $e^0 = 1$, which would give future tokens non-zero attention probabilities. $e^{-\infty} = 0$, which guarantees future tokens get exactly $0\%$ attention.

In a causal masked attention matrix, what is the value of entry $A_{1, 3}$ (Token 1 attending to Token 3)? #card
Exactly $0.0$, because position 3 is in the future relative to position 1.

## 20. Sources
- Vaswani et al. (2017) *"Attention Is All You Need"*, Section 3.2.3.
- Radford et al. (2018) *"Improving Language Understanding by Generative Pre-Training"* (GPT-1).

