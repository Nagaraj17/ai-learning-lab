# 26 - TRANSFORMER - Multi-Head Attention

## 1. The Problem
A single attention head produces **one single set of attention weights** $\mathbf{A} \in \mathbb{R}^{T \times T}$.
However, a word in a sentence often has multiple different relationships simultaneously:
- Syntactic relationship (e.g., verb attending to its subject).
- Semantic relationship (e.g., pronoun attending to its noun antecedent).
- Positional relationship (e.g., attending to immediately preceding token).

**The limitation:** Single-Head Attention averages all these different relationships into one single attention distribution, diluting specific semantic signals.

## 2. Why We Need Something New
We need **Multi-Head Attention (MHA)**, which splits the hidden dimension $d_{model}$ across $h$ independent "heads", allowing the model to jointly attend to information from different representation subspaces at different positions.

## 3. One-Line Definition
**Multi-Head Attention** runs multiple learned Q/K/V projections in parallel, performs attention in each projected subspace, concatenates the outputs, and applies an output projection $\mathbf{W}_O$.

## 4. Beginner Intuition / Mental Model
Imagine a **Detective Panel investigating a crime scene**:
- Head 1 focuses exclusively on **fingerprints** (Syntactic role).
- Head 2 focuses exclusively on **financial records** (Semantic role).
- Head 3 focuses exclusively on **timeline order** (Positional role).

Instead of one detective trying to look at everything at once, $h$ specialized detectives investigate in parallel, and their reports are combined into a final verdict!

## 5. What Came Before → What Changes Now
- **Single-Head Attention:** 1 set of projection matrices $(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V)$ of size $(d_{model} \times d_{model})$.
- **Multi-Head Attention:** $h$ parallel sets of projection matrices $(\mathbf{W}_Q^i, \mathbf{W}_K^i, \mathbf{W}_V^i)$ of size $(d_{model} \times d_k)$, concatenated and projected by $\mathbf{W}_O \in \mathbb{R}^{d_{model} \times d_{model}}$.

## 6. How It Works
1. Given sequence matrix $\mathbf{X} \in \mathbb{R}^{T \times d_{model}}$ and number of heads $h$.
2. For each head $i \in \{1, \dots, h\}$, project and compute attention:
   $$\text{head}_i = \text{Attention}(\mathbf{X} \mathbf{W}_Q^i, \mathbf{X} \mathbf{W}_K^i, \mathbf{X} \mathbf{W}_V^i) \in \mathbb{R}^{T \times d_v}$$
3. Concatenate all $h$ head outputs along the feature dimension:
   $$\text{Concat}(\text{head}_1, \dots, \text{head}_h) \in \mathbb{R}^{T \times (h \cdot d_v)}$$
4. Multiply by final output projection matrix $\mathbf{W}_O \in \mathbb{R}^{(h \cdot d_v) \times d_{model}}$:
   $$\mathbf{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}_O \in \mathbb{R}^{T \times d_{model}}$$

> **Implementation Convention:** In the original Transformer, $d_k = d_v = d_{model} / h$ was chosen so the concatenated head width equals $d_{model}$. This is a common implementation design, not a universal mathematical definition for MHA.

```
Sequence X (T x d_model)
   │
   ├──► Head 1 (W_Q1, W_K1, W_V1) ──► head_1 (T x d_k) ──┐
   ├──► Head 2 (W_Q2, W_K2, W_V2) ──► head_2 (T x d_k) ──┼──► Concat ──► (T x d_model) ──► @ W_O ──► Output (T x d_model)
   └──► Head h (W_Qh, W_Kh, W_Vh) ──► head_h (T x d_k) ──┘
```

## 7. Required Mathematics
$$\mathbf{MultiHead}(\mathbf{X}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}_O$$

where $\text{head}_i = \text{Attention}(\mathbf{X} \mathbf{W}_Q^i, \mathbf{X} \mathbf{W}_K^i, \mathbf{X} \mathbf{W}_V^i)$.

**Shape Trace:**
- Input $\mathbf{X}$: $(T \times d_{model})$
- Each $\text{head}_i$: $(T \times d_v)$
- Concatenated heads: $(T \times h \cdot d_v)$
- Output Weight $\mathbf{W}_O$: $(h \cdot d_v \times d_{model})$
- Final Output: $(T \times d_{model})$

### Symbol Table

| Symbol | Name | Plain-English Meaning |
| :--- | :--- | :--- |
| $h$ | **Number of Heads** | How many parallel attention heads run simultaneously. Each head specializes in a different relationship type (e.g., syntactic, semantic, positional). |
| $d_k, d_v$ | **Per-Head Dimensions** | The query/key and value dimensions for a single head. |
| $\mathbf{W}_Q^i, \mathbf{W}_K^i, \mathbf{W}_V^i$ | **Per-Head Projection Weights** | Each head $i$ has its own Q/K/V projection matrices. $\mathbf{W}_Q^i, \mathbf{W}_K^i \in \mathbb{R}^{d_{model} \times d_k}$, $\mathbf{W}_V^i \in \mathbb{R}^{d_{model} \times d_v}$. |
| $\text{head}_i$ | **Output of Head $i$** | The $(T \times d_v)$ contextual output from a single attention head. |
| $\text{Concat}(\dots)$ | **Concatenation** | Joins all $h$ head outputs side-by-side along the feature dimension: $h$ matrices of $(T \times d_v)$ become one $(T \times h \cdot d_v)$ matrix. |
| $\mathbf{W}_O$ | **Output Projection Matrix** | A $(h \cdot d_v \times d_{model})$ learnable weight matrix that mixes the concatenated head outputs into the final representation. This allows heads to share and combine their findings. |

> **Practical Note:** Implementations often combine all per-head projections into large $\mathbf{W}_Q$, $\mathbf{W}_K$, $\mathbf{W}_V$ matrices and reshape/split the resulting feature dimension into heads. The NumPy code below does exactly this, which is why it requires $d_{model}$ to be divisible by $h$.

## 8. Complete Worked Example
Let $T = 2$, $d_{model} = 4$, $h = 2$ heads $\implies d_k = d_v = 4 / 2 = 2$.

Input $\mathbf{X} \in \mathbb{R}^{2 \times 4}$.

- **Head 1:** Projects $\mathbf{X}$ to $\mathbf{Q}_1, \mathbf{K}_1, \mathbf{V}_1 \in \mathbb{R}^{2 \times 2}$. Computes $\text{head}_1 \in \mathbb{R}^{2 \times 2}$.
- **Head 2:** Projects $\mathbf{X}$ to $\mathbf{Q}_2, \mathbf{K}_2, \mathbf{V}_2 \in \mathbb{R}^{2 \times 2}$. Computes $\text{head}_2 \in \mathbb{R}^{2 \times 2}$.

Concatenate along columns:

$$\text{Concat}(\text{head}_1, \text{head}_2) = \begin{bmatrix} [\text{head}_1 \text{ row 0}] & [\text{head}_2 \text{ row 0}] \\ [\text{head}_1 \text{ row 1}] & [\text{head}_2 \text{ row 1}] \end{bmatrix} \in \mathbb{R}^{2 \times 4}$$

Multiply by $\mathbf{W}_O \in \mathbb{R}^{4 \times 4}$ to get final Output of shape $(2, 4)$.

## 9. Math → Code Mapping
```python
import numpy as np

class MultiHeadAttentionNumPy:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = np.random.randn(d_model, d_model) * 0.1
        self.W_K = np.random.randn(d_model, d_model) * 0.1
        self.W_V = np.random.randn(d_model, d_model) * 0.1
        self.W_O = np.random.randn(d_model, d_model) * 0.1

    def forward(self, X, mask=None):
        T, d = X.shape
        # Linear projections
        Q = X @ self.W_Q # (T, d_model)
        K = X @ self.W_K # (T, d_model)
        V = X @ self.W_V # (T, d_model)

        # Split heads: (T, h, d_k) -> transpose to (h, T, d_k)
        Q_heads = Q.reshape(T, self.num_heads, self.d_k).transpose(1, 0, 2)
        K_heads = K.reshape(T, self.num_heads, self.d_k).transpose(1, 0, 2)
        V_heads = V.reshape(T, self.num_heads, self.d_k).transpose(1, 0, 2)

        # Scaled dot-product attention per head
        scores = (Q_heads @ K_heads.transpose(0, 2, 1)) / np.sqrt(self.d_k) # (h, T, T)
        if mask is not None:
            scores = np.where(mask == 1, scores, -1e9)

        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        A = exp_s / np.sum(exp_s, axis=-1, keepdims=True) # (h, T, T)

        heads_out = A @ V_heads # (h, T, d_k)
        # Concatenate heads: (T, h * d_k) = (T, d_model)
        concat = heads_out.transpose(1, 0, 2).reshape(T, self.d_model)

        output = concat @ self.W_O # (T, d_model)
        return output, A
```

### Understanding the Reshape → Transpose Pipeline

The trickiest part of the Multi-Head code is how we split one large Q/K/V matrix into $h$ separate heads **without a for-loop**. Here's what each line does:

```python
# Step 1: Q has shape (T, d_model) = (T, h * d_k)
# We reshape it to (T, h, d_k) — this "splits" the features into h groups of d_k
Q_heads = Q.reshape(T, self.num_heads, self.d_k)
# Shape: (T, h, d_k) — but we need (h, T, d_k) for batched matmul

# Step 2: Transpose axes so heads become the first dimension
Q_heads = Q_heads.transpose(1, 0, 2)
# Shape: (h, T, d_k) — now each head[i] is a (T, d_k) matrix
# This is like having h separate Q matrices stacked in a batch!
```

**Why do we need this?** Because `Q_heads @ K_heads.T` computes `h` independent attention score matrices $(T \times T)$ simultaneously as a single batched operation — much faster than running a Python loop $h$ times.

After computing attention outputs for each head `(h, T, d_k)`, we reverse the process:
```python
# Transpose back: (h, T, d_k) → (T, h, d_k)
# Then reshape: (T, h, d_k) → (T, h * d_k) = (T, d_model)
concat = heads_out.transpose(1, 0, 2).reshape(T, self.d_model)
```

## 10. Experiments / What-If Questions
- **Does Multi-Head Attention increase total computational FLOPs compared to Single-Head Attention of dimension $d_{model}$?**
  The original Transformer states that with the design choice $d_k = d_v = d_{model} / h$, the total computational cost is similar to single-head attention with full dimensionality. Equal splitting prevents the attention computation from becoming $h$ full-width attention operations. However, there are still multiple projections, concatenation, output projection, and memory/implementation overheads.

## 11. Common Misunderstandings
- **Misunderstanding:** Each attention head processes a different subset of words in the sentence.
- **Correction:** Every head processes **all $T$ words in the sequence**, but in a different feature subspace of dimension $d_k$.

## 12. Limitations and Trade-Offs
- **Memory Scaling:** Each head conceptually has a $T \times T$ attention matrix, so storing all attention weights can scale with $h \times T \times T$ (although optimized kernels may avoid explicitly materializing every intermediate).
- The common reshape-based implementation (like the NumPy code above) requires $d_{model}$ to be divisible by $h$, but this is not a universal mathematical requirement for MHA.
- Pruning experiments show that some attention heads become redundant during training and can be pruned without degrading model performance.

## 13. Where It Appears in the Current Assignment
In **Week 3 Assignment**, Multi-Head Attention is introduced conceptually and implemented as an optional stretch exercise.

## 14. Where It Appears in Modern AI Systems
MHA is foundational; later architectures sometimes modify how query/key/value heads are shared for efficiency (e.g., Multi-Query Attention or Grouped-Query Attention).

## 15. Connection to the Next Concept
Multi-Head Causal Attention produces contextual representations for sequence matrix $\mathbf{X}$. To build a complete Transformer block, we combine Multi-Head Attention with Layer Normalization, Residual Connections, and FeedForward Networks in **Week 4 (Transformer Block)**.

## 16. Teach-Back and Small Application Exercise
If $d_{model} = 768$ and the model uses $h = 12$ heads:
1. What is the head dimension $d_k$ for each head?
2. What is the shape of attention scores for each head in a sequence of length $T = 16$?

## 17. Quick Revision Summary
- Multi-Head Attention splits $d_{model}$ across $h$ parallel heads of size $d_k = d_{model} / h$.
- Allows the model to attend to multiple representation subspaces simultaneously.
- Output is concatenated and projected by $\mathbf{W}_O$.

## 18. My Understanding
*Fill in your own notes on how Multi-Head Attention enables parallel representation learning.*

## 19. Flashcards
Is $d_k = d_{model} / h$ a universal mathematical requirement of Multi-Head Attention? #card
No. It is a common implementation convention (used in the original Transformer) to ensure the concatenated output width equals $d_{model}$, preventing the total computation from growing too large.

Why does Multi-Head Attention with $d_k = d_{model} / h$ have similar computational cost to a single head of full dimension $d_{model}$? #card
Because each head projects into a smaller dimension $d_k$. Equal splitting prevents the attention computation from becoming $h$ full-width attention operations, keeping the total dot-product cost similar.

## 20. Sources
- Vaswani et al. (2017) *"Attention Is All You Need"*, Section 3.2.2.
- Alammar, J. & Grootendorst, M. [Hands-On Large Language Models.md](file:///c:/Users/Nagar/source/repos/ai-learning-lab/resources/references/Hands-On%20Large%20Language%20Models.md), Chapter 3.

