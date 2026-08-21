# 31 - TRANSFORMER - Feed-Forward Networks and Representation Expansion

## 1. The Problem
Multi-Head Attention (MHA) allows tokens to gather context from other tokens across a sequence. However, MHA consists entirely of **linear combinations** (weighted matrix averages $A V$).

If a Transformer consisted *only* of MHA layers, it would be incapable of learning complex non-linear relationships. Attention decides **where to look** and **what context to gather**, but it lacks the capacity to process, transform, and store factual knowledge about those gathered features.

---

## 2. Why We Need Something New
We need a component that:
1. Applies **non-linear activation functions** (like ReLU or GELU) to transform feature representations.
2. Operates on **each token position independently and identically** (position-wise processing).
3. Expands the hidden representation space ($d_{\text{ff}} = 4 \times d_{\text{model}}$) to allow key-value memory retrieval before projecting back to $d_{\text{model}}$.

That component is the **Position-Wise Feed-Forward Network (FFN)**.

---

## 3. One-Line Definition
The **Position-Wise Feed-Forward Network (FFN)** is a two-layer fully connected neural network applied independently to each token position, projecting features up to a higher-dimensional hidden space ($4 \times d_{\text{model}}$) with a non-linear activation before projecting back to $d_{\text{model}}$.

---

## 4. Beginner Intuition / Mental Model
- **Attention Sub-layer**: The **Researcher** who goes out to the library and gathers relevant reference books ($X \rightarrow A V$).
- **Feed-Forward Sub-layer (FFN)**: The **Analyst** who takes the gathered books, sits down at their desk, thinks deeply, applies non-linear reasoning, and writes down the synthesized conclusion!

---

## 5. What Came Before $\rightarrow$ What Changes Now

| Aspect | Multi-Head Attention Sub-layer | Feed-Forward Network Sub-layer |
| :--- | :--- | :--- |
| **Primary Goal** | **Context Gathering** (where to look across sequence). | **Feature Processing** (what to do with gathered context). |
| **Cross-Token Mixing** | **Yes**: Mixes information across sequence length $T$. | **No**: Operates on each token position independently. |
| **Linearity** | Linear combinations ($A V W_O$). | **Non-Linear**: Uses ReLU / GELU activation functions. |
| **Hidden Dimension** | Standard $d_{\text{model}}$. | **Expanded**: $d_{\text{ff}} = 4 \times d_{\text{model}}$ memory bank. |

---

## 6. How It Works
For each token vector $x \in \mathbb{R}^{d_{\text{model}}}$:

1. **Linear Expansion**: Project vector from $d_{\text{model}}$ to $d_{\text{ff}} = 4 \times d_{\text{model}}$:
   $$h_1 = x W_1 + b_1$$
2. **Non-Linear Activation**: Apply ReLU (or GELU/SwiGLU):
   $$a_1 = \max(0, h_1)$$
3. **Linear Compression**: Project vector back from $d_{\text{ff}}$ to $d_{\text{model}}$:
   $$y = a_1 W_2 + b_2$$

---

## 7. Required Mathematics

### Formulas:
$$\text{FFN}(x) = \text{ReLU}\left(x W_1 + b_1\right) W_2 + b_2$$

### Symbol-by-Symbol Breakdown:
- $x$: Input feature vector for one token position with shape $(d_{\text{model}},)$.
- $W_1$: First weight matrix with shape $(d_{\text{model}}, d_{\text{ff}})$.
- $b_1$: First bias vector with shape $(d_{\text{ff}},)$.
- $\text{ReLU}$: Rectified Linear Unit activation function ($\max(0, z)$).
- $W_2$: Second weight matrix with shape $(d_{\text{ff}}, d_{\text{model}})$.
- $b_2$: Second bias vector with shape $(d_{\text{model}},)$.

### Tensor Shape Trace:
- Input $X$: $(B, T, d_{\text{model}})$
- After $W_1 + b_1$: $(B, T, d_{\text{ff}})$
- After $\text{ReLU}$: $(B, T, d_{\text{ff}})$
- After $W_2 + b_2$: $(B, T, d_{\text{model}})$

---

## 8. Complete Worked Example

Let input vector be $x = [1.0, 2.0]$ ($d_{\text{model}}=2, d_{\text{ff}}=4$):

1. **Projection $W_1$ ($2 \rightarrow 4$)**:
   Let $W_1 = \begin{bmatrix} 1 & 0 & -1 & 2 \\ 0 & 1 & -1 & 1 \end{bmatrix}, b_1 = [0, 0, 0, 0]$.
   $$h_1 = x W_1 = [1.0(1)+2.0(0), \; 1.0(0)+2.0(1), \; 1.0(-1)+2.0(-1), \; 1.0(2)+2.0(1)] = [1.0, \; 2.0, \; -3.0, \; 4.0]$$

2. **ReLU Activation**:
   $$a_1 = \text{ReLU}([1.0, 2.0, -3.0, 4.0]) = [1.0, \; 2.0, \; 0.0, \; 4.0]$$

3. **Projection $W_2$ ($4 \rightarrow 2$)**:
   Let $W_2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 1 \end{bmatrix}, b_2 = [0, 0]$.
   $$y = a_1 W_2 = [1.0(1)+2.0(0)+0(1)+4(0), \; 1.0(0)+2.0(1)+0(1)+4(1)] = [1.0, \; 6.0]$$

The input vector $[1.0, 2.0]$ was non-linearly transformed into $[1.0, 6.0]$!

---

## 9. Math $\rightarrow$ Code Mapping

```python
class FeedForwardNumPy:
    def __init__(self, d_model, d_ff, rng):
        limit1 = np.sqrt(6.0 / (d_model + d_ff))
        self.W1 = rng.uniform(-limit1, limit1, (d_model, d_ff)).astype(np.float32)
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        limit2 = np.sqrt(6.0 / (d_ff + d_model))
        self.W2 = rng.uniform(-limit2, limit2, (d_ff, d_model)).astype(np.float32)
        self.b2 = np.zeros(d_model, dtype=np.float32)

    def forward(self, x):
        h1 = np.matmul(x, self.W1) + self.b1
        a1 = np.maximum(0, h1)  # ReLU
        out = np.matmul(a1, self.W2) + self.b2
        self.cache = (x, h1, a1)
        return out
```

---

## 10. Experiments / What-If Questions
- **Why expand to $4 \times d_{\text{model}}$ instead of staying at $1 \times d_{\text{model}}$?** Research (Geva et al., 2021) showed $W_1$ acts as a key-value memory bank. Expanding to $4 \times$ creates 4x more pattern detectors to store facts.
- **What if we remove FFN?** In our Week 5 benchmark, Model D-no-FFN dropped performance compared to Model D-1, confirming that non-linear feature expansion is necessary to process complex step combinations.

---

## 11. Common Misunderstandings
- ❌ *Misconception*: "FFN combines tokens across the sequence length $T$."
  - ✅ **Correction**: No! FFN operates on each token position independently. There is zero interaction between position $i$ and position $j$ inside the FFN.

---

## 12. Limitations and Trade-Offs
- **Parameter Count**: FFN layers account for **~66% of total trainable parameters** in a Transformer block!

---

## 13. Where It Appears in the Current Assignment
Used in every Transformer block in **Week 5**: $d_{\text{model}}=24 \rightarrow d_{\text{ff}}=96 \rightarrow d_{\text{model}}=24$.

---

## 14. Where It Appears in Modern AI Systems
- **GPT-3 / GPT-4**: Standard FFN with GELU activation.
- **LLaMA-3 / Mistral**: Uses **SwiGLU** (Swish Gated Linear Unit) FFN layers.
- **Mixtral 8x7B (MoE)**: Replaces standard FFN with 8 expert FFN layers per token.

---

## 15. Connection to the Next Concept
With Attention (Context) and FFN (Memory) combined into one block, we can now **Stack Multiple Blocks** to build deep LLM representations!

---

## 16. Teach-Back and Small Application Exercise
**Exercise**: Why does Multi-Head Attention need an FFN sub-layer right after it?

---

## 17. Quick Revision Summary
- FFN is a 2-layer position-wise network ($\text{ReLU}(X W_1 + b_1) W_2 + b_2$).
- Expands dimension by $4 \times d_{\text{model}}$ to act as an associative memory bank.
- Applies non-linear activation to process gathered context.

---

## 18. My Understanding

```markdown
The Feed-Forward Network (FFN) expands each token's vector by 4x, applies a non-linear activation like ReLU to extract feature patterns, and projects it back to d_model. While attention gathers context from other tokens, FFN decides what to do with that context and stores factual memory.
```

---

## 19. Flashcards

**Front**: What percentage of parameters in a Transformer block belong to the FFN?  
**Back**: Approximately ~66% (two-thirds) of total parameters.

**Front**: Does the FFN combine information across different token positions?  
**Back**: No. FFN is position-wise; it operates on each token position vector independently and identically.

---

## 20. Sources
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
- Geva, M., et al. (2021). *Transformer Feed-Forward Layers Are Key-Value Memories*. EMNLP.
