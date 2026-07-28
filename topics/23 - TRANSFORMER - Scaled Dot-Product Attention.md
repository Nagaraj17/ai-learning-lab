# 23 - TRANSFORMER - Scaled Dot-Product Attention

## 1. The Problem
When calculating dot products $\mathbf{S} = \mathbf{Q} \mathbf{K}^\top$ for large vector dimensions $d_k$ (e.g. $d_k = 64$ or $128$), the magnitude of the dot products grows large: $\text{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k$.
When unscaled large values (e.g. scores of $+50$ vs $+2$) are passed into Softmax, Softmax pushes probability weights to extreme $1.0$ and $0.0$ values. 
**The limitation:** Softmax gradients in those extreme regions become almost $0.0$ ($\text{Softmax}'(z) \to 0$), causing **vanishing gradients** during backpropagation!

## 2. Why We Need Something New
We need a scaling factor that divides raw dot-product scores by $\sqrt{d_k}$ to keep variance equal to $1.0$, preserving healthy non-zero gradients during Softmax backpropagation.

## 3. One-Line Definition
**Scaled Dot-Product Attention** is an attention mechanism that computes pairwise relevance scores using matrix multiplication $\mathbf{Q} \mathbf{K}^\top$, scales the scores by $\frac{1}{\sqrt{d_k}}$, and applies Softmax to calculate normalized attention weights over Value matrix $\mathbf{V}$.

## 4. Beginner Intuition / Mental Model
Imagine a volume knob turned up to 100 on a stereo speaker — the music becomes distorted and clipped.
Scaling by $\frac{1}{\sqrt{d_k}}$ is like a **Volume Limiter**: it turns down extreme score spikes back into a clean range so Softmax can calculate smooth, informative probability percentages.

## 5. What Came Before → What Changes Now
- **Additive Attention (Bahdanau et al., 2014):** Computed score $e_{i, j} = \mathbf{v}_a^\top \tanh(\mathbf{W}_a \mathbf{q}_i + \mathbf{U}_a \mathbf{k}_j)$ using a feedforward network. Slower $O(T^2 \cdot d)$ operations.
- **Scaled Dot-Product Attention (Vaswani et al., 2017):** Computes score matrix $\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}$ using highly optimized GPU matrix multiplication.

## 6. How It Works
1. Compute raw pairwise score matrix: $\mathbf{S} = \mathbf{Q} \mathbf{K}^\top \in \mathbb{R}^{T \times T}$.
2. Scale scores by square root of key dimension: $\mathbf{S}_{\text{scaled}} = \frac{\mathbf{S}}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}$.
3. Apply row-wise Softmax to compute attention weights: $\mathbf{A} = \text{Softmax}(\mathbf{S}_{\text{scaled}}) \in \mathbb{R}^{T \times T}$.
4. Compute weighted sum of Values: $\mathbf{H} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{T \times d_v}$.

> **Softmax Refresher (from Topic 08):**
> Softmax converts a row of raw scores $[z_1, z_2, \dots, z_n]$ into a probability distribution:
> $$P_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$
> Each output $P_i$ is between $0.0$ and $1.0$, and all outputs in the row sum to exactly $1.0$.
> In attention, Softmax is applied **independently to each row** of the $(T \times T)$ score matrix.

```
Q (T x d_k) ──┐
              ├──► MatMul (Q @ K.T) ──► S (T x T) ──► Scale (/ sqrt(d_k)) ──► Softmax ──► A (T x T) ──┐
K (T x d_k) ──┘                                                                                        ├──► MatMul (A @ V) ──► H (T x d_v)
V (T x d_v) ───────────────────────────────────────────────────────────────────────────────────────────┘
```

## 7. Required Mathematics
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} \right) \mathbf{V}$$

**Shape Trace:**
- $\mathbf{Q}: (T \times d_k)$
- $\mathbf{K}^\top: (d_k \times T)$
- $\mathbf{Q} \mathbf{K}^\top: (T \times T)$
- $\frac{1}{\sqrt{d_k}}$: Scalar constant (Shape unchanged: $T \times T$)
- $\text{Softmax}(\dots): (T \times T)$ (Row sums equal $1.0$)
- $\mathbf{V}: (T \times d_v)$
- Output $\mathbf{H}: (T \times T) \cdot (T \times d_v) = (T \times d_v)$

### Symbol Table

| Symbol | Name | Plain-English Meaning |
| :--- | :--- | :--- |
| $\mathbf{Q}$ | **Query Matrix** | $(T \times d_k)$ — each row is a token's "search request" vector. |
| $\mathbf{K}$ | **Key Matrix** | $(T \times d_k)$ — each row is a token's "index tag" vector. |
| $\mathbf{K}^\top$ | **Transposed Key Matrix** | $(d_k \times T)$ — K flipped so inner dimensions align for $\mathbf{Q} \mathbf{K}^\top$. |
| $\mathbf{V}$ | **Value Matrix** | $(T \times d_v)$ — each row is a token's "content" to be retrieved. |
| $\mathbf{S}$ | **Raw Score Matrix** | $(T \times T)$ — result of $\mathbf{Q} \mathbf{K}^\top$. Entry $S_{i,j}$ is the raw dot-product similarity of token $i$'s Query and token $j$'s Key. |
| $d_k$ | **Key/Query Dimension** | Number of features per Q/K vector. Determines how much dot products can grow. |
| $\sqrt{d_k}$ | **Scaling Factor** | The square root of $d_k$. We divide raw scores by this to keep variance ≈ 1.0. For $d_k = 64$: $\sqrt{64} = 8$. |
| $\mathbf{S}_{\text{scaled}}$ | **Scaled Score Matrix** | $\frac{\mathbf{S}}{\sqrt{d_k}}$ — scores brought into a moderate range before Softmax. |
| $\mathbf{A}$ | **Attention Weight Matrix** | $(T \times T)$ — result of row-wise Softmax on scaled scores. Each row sums to $1.0$. |
| $\mathbf{H}$ | **Contextual Output** | $(T \times d_v)$ — result of $\mathbf{A} \mathbf{V}$. Each token's updated representation. |

## 8. Complete Worked Example
Let $T = 2$, $d_k = 2$, $d_v = 2$. Therefore $\sqrt{d_k} = \sqrt{2} \approx 1.414$.

Let $\mathbf{Q} = \begin{bmatrix} 1.0 & 4.0 \\ 3.0 & 0.0 \end{bmatrix}$, $\mathbf{K} = \begin{bmatrix} 2.0 & 1.0 \\ 0.0 & 3.0 \end{bmatrix}$, $\mathbf{V} = \begin{bmatrix} 1.0 & 3.0 \\ 3.0 & 3.0 \end{bmatrix}$.

1. **Raw Dot Product $\mathbf{S} = \mathbf{Q} \mathbf{K}^\top$:**

$$\mathbf{S} = \begin{bmatrix} 1 & 4 \\ 3 & 0 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 1 & 3 \end{bmatrix} = \begin{bmatrix} (1\cdot2 + 4\cdot1) & (1\cdot0 + 4\cdot3) \\ (3\cdot2 + 0\cdot1) & (3\cdot0 + 0\cdot3) \end{bmatrix} = \begin{bmatrix} 6.0 & 12.0 \\ 6.0 & 0.0 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$

2. **Scale by $\sqrt{d_k} = 1.414$:**

$$\mathbf{S}_{\text{scaled}} = \frac{1}{1.414} \begin{bmatrix} 6.0 & 12.0 \\ 6.0 & 0.0 \end{bmatrix} \approx \begin{bmatrix} 4.24 & 8.49 \\ 4.24 & 0.00 \end{bmatrix}$$

3. **Row-wise Softmax $\mathbf{A} = \text{Softmax}(\mathbf{S}_{\text{scaled}})$:**

   **Row 0 — full arithmetic** (so you can verify yourself):
   - Inputs: $[4.24, 8.49]$
   - Exponentiate each: $e^{4.24} \approx 69.4$, $e^{8.49} \approx 4875.1$
   - Sum of exponentials: $69.4 + 4875.1 = 4944.5$
   - Divide each by sum: $\frac{69.4}{4944.5} \approx 0.014$, $\frac{4875.1}{4944.5} \approx 0.986$
   - Result: $[0.014, 0.986]$ ✓ (sums to $1.0$)

   > **Notice:** The large gap between $4.24$ and $8.49$ causes Softmax to push almost all probability ($98.6\%$) to the second token. This is why scaling matters — without it, the gaps would be even larger and Softmax would output a hard $[0.0, 1.0]$.

   **Row 1:** $\text{Softmax}([4.24, 0.00])$:
   - $e^{4.24} \approx 69.4$, $e^{0.00} = 1.0$
   - Sum: $70.4$
   - Result: $[\frac{69.4}{70.4}, \frac{1.0}{70.4}] \approx [0.986, 0.014]$

$$\mathbf{A} = \begin{bmatrix} 0.014 & 0.986 \\ 0.986 & 0.014 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$

4. **Weighted Sum $\mathbf{H} = \mathbf{A} \mathbf{V}$:**

$$\mathbf{H} = \begin{bmatrix} 0.014 & 0.986 \\ 0.986 & 0.014 \end{bmatrix} \begin{bmatrix} 1.0 & 3.0 \\ 3.0 & 3.0 \end{bmatrix} = \begin{bmatrix} (0.014\cdot1 + 0.986\cdot3) & (0.014\cdot3 + 0.986\cdot3) \\ (0.986\cdot1 + 0.014\cdot3) & (0.986\cdot3 + 0.014\cdot3) \end{bmatrix} \approx \begin{bmatrix} 2.97 & 3.00 \\ 1.03 & 3.00 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$

## 9. Math → Code Mapping
```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    
    # 1. Raw scores S = Q @ K.T
    S = Q @ K.T
    
    # 2. Scale by sqrt(d_k)
    S_scaled = S / np.sqrt(d_k)
    
    # 3. Row-wise Softmax
    exp_S = np.exp(S_scaled - np.max(S_scaled, axis=-1, keepdims=True))
    A = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
    
    # 4. Weighted sum H = A @ V
    H = A @ V
    return H, A
```

## 10. Experiments / What-If Questions
- **What happens if we remove the scaling factor $\frac{1}{\sqrt{d_k}}$?**
  For large $d_k$ (e.g., $d_k = 128$), dot products reach $+80$, Softmax outputs become one-hot $1.0$ and $0.0$, and gradients vanish during backpropagation ($\text{Softmax}' \to 0$).

## 11. Common Misunderstandings
- **Misunderstanding:** Scaling by $\sqrt{d_k}$ changes the relative ranking order of the scores.
- **Correction:** Division by a positive constant $c = \sqrt{d_k}$ is monotonic! The largest raw dot product remains the largest score after scaling. It only scales down variance so Softmax gradients don't vanish.

## 12. Limitations and Trade-Offs
Dot-product attention assumes $Q$ and $K$ lie in compatible inner-product vector spaces ($d_k$). The $O(T^2)$ matrix multiplication becomes a memory bottleneck for extremely long sequences ($T > 32,000$).

## 13. Where It Appears in the Current Assignment
In **Week 3 Assignment**, you will implement `scaled_dot_product_attention(Q, K, V)` in NumPy and verify its shape transformations.

## 14. Where It Appears in Modern AI Systems
Scaled Dot-Product Attention is the foundational mathematical equation of all modern Transformer models (Vaswani et al., 2017).

## 15. Connection to the Next Concept
In a complete Self-Attention layer, how do we prevent tokens from peeking into future positions during autoregressive next-token prediction? That requires **Causal Masking** (`25 - TRANSFORMER - Causal Masking.md`).

## 16. Teach-Back and Small Application Exercise
If $d_k = 64$:
1. What is the scaling factor $\sqrt{d_k}$?
2. If raw dot product is $32.0$, what is the scaled dot product score?

## 17. Quick Revision Summary
- Scaled Dot-Product Attention formula: $\text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}$.
- Scaling by $\sqrt{d_k}$ prevents vanishing gradients in Softmax.

## 18. My Understanding
*Fill in your own notes on how scaling by $\sqrt{d_k}$ protects Softmax gradients.*

## 19. Flashcards
What is the mathematical equation for Scaled Dot-Product Attention? #card
$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$

Why do we scale the dot products by $\frac{1}{\sqrt{d_k}}$? #card
To prevent dot product magnitudes from growing too large for high dimensions $d_k$, which would push Softmax into extreme regions with near-zero gradients.

## 20. Sources
- Vaswani et al. (2017) *"Attention Is All You Need"*, Section 3.2.1.
- Goodfellow, I., Bengio, Y., & Courville, A. [Deep Learning.md](file:///c:/Users/Nagar/source/repos/ai-learning-lab/resources/references/Deep%20Learning.md), Chapter 6 (Softmax Vanishing Gradients).

