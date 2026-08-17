# 31 - TRANSFORMER - Feed-Forward Networks and Representation Expansion

## 1. The Problem
Multi-Head Attention (MHA) allows tokens to gather context from other tokens across a sequence. However, MHA consists entirely of **linear combinations** (weighted matrix averages $A @ V$).

If a Transformer consisted *only* of MHA layers, it would be incapable of learning complex non-linear relationships. Attention decides **where to look** and **what context to gather**, but it lacks the capacity to process, transform, and store factual knowledge about those gathered features.

---

## 2. Why We Need Something New: Position-Wise Feed-Forward Network
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
- **Attention Sub-layer**: The **Researcher** who goes out to the library and gathers relevant reference books ($X \rightarrow A @ V$).
- **Feed-Forward Sub-layer (FFN)**: The **Analyst** who takes the gathered books, sits down at their desk, thinks deeply, applies non-linear reasoning, and writes down the synthesized conclusion!

---

## 5. Why Expand to $4 \times d_{\text{model}}$ before Shrinking?

In standard Transformers:
- Input dimension: $d_{\text{model}} = 16$ (or $512$ / $4096$).
- Hidden FFN dimension: $d_{\text{ff}} = 4 \times d_{\text{model}} = 64$ (or $2048$ / $16384$).

### Why expand by $4\times$?
Recent research (Geva et al., 2021) showed that the first FFN linear matrix ($W_1$) acts as an **associative Key-Value memory bank**:
1. **Expansion ($W_1 \in \mathbb{R}^{d_{\text{model}} \times 4 d_{\text{model}}}$)**: Projects the token vector into a vast 64-dimensional space where thousands of pattern detectors (keys) fire non-linearly via ReLU.
2. **Shrinking ($W_2 \in \mathbb{R}^{4 d_{\text{model}} \times d_{\text{model}}}$)**: Combines the active pattern detectors back into a refined 16-dimensional representation vector!

---

## 6. The Mathematical Formulas

For each token position vector $x \in \mathbb{R}^{d_{\text{model}}}$:

$$\text{FFN}(x) = \text{ReLU}\left(x W_1 + b_1\right) W_2 + b_2$$

### Symbol-by-Symbol Breakdown:
- $x$: Token feature vector of shape $(1, d_{\text{model}})$.
- $W_1$: First weight matrix of shape $(d_{\text{model}}, 4 \cdot d_{\text{model}})$.
- $b_1$: First bias vector of shape $(4 \cdot d_{\text{model}},)$.
- $\text{ReLU}(z) = \max(0, z)$: Rectified Linear Unit activation function (zeroes out negative activations, introducing non-linearity).
- $W_2$: Second weight matrix of shape $(4 \cdot d_{\text{model}}, d_{\text{model}})$.
- $b_2$: Second bias vector of shape $(d_{\text{model}},)$.

---

## 7. Complete Worked Example (Small Numbers)

Let $d_{\text{model}} = 2$ and $d_{\text{ff}} = 4$.
Input token vector $x = [1.0, \; 2.0]$.

Suppose:
$$W_1 = \begin{bmatrix} 1 & -1 & 2 & 0 \\ 0 & 1 & -1 & 2 \end{bmatrix}, \quad b_1 = [0, 0, 0, 0]$$
$$W_2 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad b_2 = [0, 0]$$

1. **First Linear Projection ($x W_1$)**:
   $$z = [1.0, 2.0] \begin{bmatrix} 1 & -1 & 2 & 0 \\ 0 & 1 & -1 & 2 \end{bmatrix} = [1.0, \; 1.0, \; 0.0, \; 4.0]$$

2. **Non-Linear Activation ($\text{ReLU}(z)$)**:
   $$\text{h} = \text{ReLU}([1.0, 1.0, 0.0, 4.0]) = [1.0, \; 1.0, \; 0.0, \; 4.0]$$

3. **Second Linear Projection ($h W_2$)**:
   $$\text{FFN}(x) = [1.0, 1.0, 0.0, 4.0] \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 1 \end{bmatrix} = [1.0, \; 5.0]$$

Result: The input $[1.0, 2.0]$ was non-linearly transformed into $[1.0, 5.0]$!

---

## 8. Python / NumPy Implementation

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def feed_forward_network(x, W1, b1, W2, b2):
    """
    Computes Position-Wise Feed-Forward Network.
    x shape: (Batch, Seq_Len, d_model)
    W1 shape: (d_model, 4 * d_model)
    W2 shape: (4 * d_model, d_model)
    """
    # 1. Expand dimension to 4 * d_model & apply ReLU
    h = relu(x @ W1 + b1) # (Batch, Seq_Len, 4 * d_model)
    
    # 2. Project back to d_model
    out = h @ W2 + b2      # (Batch, Seq_Len, d_model)
    
    cache = (x, h, W1, b1, W2, b2)
    return out, cache

# Test Setup
d_model = 16
d_ff = 64 # 4 * 16

W1 = np.random.randn(d_model, d_ff) * 0.1
b1 = np.zeros(d_ff)
W2 = np.random.randn(d_ff, d_model) * 0.1
b2 = np.zeros(d_model)

x_test = np.random.randn(2, 4, d_model)
ffn_out, _ = feed_forward_network(x_test, W1, b1, W2, b2)
print("FFN Output Shape:", ffn_out.shape) # Result: (2, 4, 16)
```

---

## 9. My Understanding

```markdown
While Multi-Head Attention gathers context across tokens, the Position-Wise Feed-Forward Network (FFN) processes each token independently. It expands the feature dimension by 4x into a hidden space where non-linear activations (ReLU/GELU) act as a key-value memory bank before projecting back to d_model.
```

---

## 10. Flashcards

**Front**: What is the primary role of the Feed-Forward Network (FFN) in a Transformer block?  
**Back**: The FFN introduces non-linear feature transformations and memory retrieval for each token position independently, complementing MHA's linear token-mixing operation.

**Front**: Why does the FFN expand the dimension to $4 \times d_{\text{model}}$ before projecting back to $d_{\text{model}}$?  
**Back**: Expanding the hidden dimension creates a high-dimensional sparse activation space where pattern-matching keys can fire non-linearly (functioning as a memory bank).

---

## 11. Sources
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
- Geva, M., et al. (2021). *Transformer Feed-Forward Layers Are Key-Value Memories*. EMNLP.
