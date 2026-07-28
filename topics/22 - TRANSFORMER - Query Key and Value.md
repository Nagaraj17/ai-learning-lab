# 22 - TRANSFORMER - Query Key and Value

## 1. The Problem
If we compare raw sequence vectors $\mathbf{X}$ directly against themselves ($\mathbf{X} \mathbf{X}^\top$), a token has to play three conflicting roles simultaneously using the exact same numbers:
1. Asking what it is looking for.
2. Advertising what information it holds.
3. Supplying its actual content.

Using one vector for all three roles prevents the model from learning asymmetric relationships (e.g. a verb searching for a noun, where the verb's search criteria differ from its content).

## 2. Why We Need Something New
We need separate, trainable linear projections that transform the single input representation $\mathbf{X}$ into three distinct functional roles: **Query ($Q$)**, **Key ($K$)**, and **Value ($V$)**.

## 3. One-Line Definition
**Query ($\mathbf{Q}$)**, **Key ($\mathbf{K}$)**, and **Value ($\mathbf{V}$)** are linear projections of the sequence matrix $\mathbf{X}$ created by multiplying $\mathbf{X}$ by three separate weight matrices ($\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$).

## 4. Beginner Intuition / Mental Model
Think of a **Database / Search Engine System**:
- **Query ($\mathbf{Q}$):** The search term you type into Google (*"What information am I looking for?"*).
- **Key ($\mathbf{K}$):** The index tags / titles of all webpage articles in the database (*"What category of information do I contain?"*).
- **Value ($\mathbf{V}$):** The actual content / text of the articles (*"Here is my full underlying information."*).

Matching Query against Key finds which articles to read; multiplying by Value retrieves the content!

## 5. What Came Before → What Changes Now
- **Before:** Single representation vector $\mathbf{x}_i$ per token.
- **Now:** Three projected vectors per token: $\mathbf{q}_i = \mathbf{x}_i \mathbf{W}_Q$, $\mathbf{k}_i = \mathbf{x}_i \mathbf{W}_K$, $\mathbf{v}_i = \mathbf{x}_i \mathbf{W}_V$.

## 6. How It Works
1. Take sequence matrix $\mathbf{X} \in \mathbb{R}^{T \times d_{model}}$.
2. Multiply by projection matrix $\mathbf{W}_Q \in \mathbb{R}^{d_{model} \times d_k}$ to get $\mathbf{Q} \in \mathbb{R}^{T \times d_k}$.
3. Multiply by projection matrix $\mathbf{W}_K \in \mathbb{R}^{d_{model} \times d_k}$ to get $\mathbf{K} \in \mathbb{R}^{T \times d_k}$.
4. Multiply by projection matrix $\mathbf{W}_V \in \mathbb{R}^{d_{model} \times d_v}$ to get $\mathbf{V} \in \mathbb{R}^{T \times d_v}$.

```
                 ┌──► Query Projection W_Q  ──► Q = X W_Q  (T x d_k)
Sequence X  ─────┼──► Key Projection W_K    ──► K = X W_K  (T x d_k)
 (T x d)         └──► Value Projection W_V  ──► V = X W_V  (T x d_v)
```

## 7. Required Mathematics
For sequence matrix $\mathbf{X} \in \mathbb{R}^{T \times d_{model}}$:

$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q \in \mathbb{R}^{T \times d_k}$$
$$\mathbf{K} = \mathbf{X} \mathbf{W}_K \in \mathbb{R}^{T \times d_k}$$
$$\mathbf{V} = \mathbf{X} \mathbf{W}_V \in \mathbb{R}^{T \times d_v}$$

**Shape Trace:**
- Input $\mathbf{X}$: $(T \times d_{model})$
- Weight $\mathbf{W}_Q$: $(d_{model} \times d_k) \implies \mathbf{Q}: (T \times d_k)$
- Weight $\mathbf{W}_K$: $(d_{model} \times d_k) \implies \mathbf{K}: (T \times d_k)$
- Weight $\mathbf{W}_V$: $(d_{model} \times d_v) \implies \mathbf{V}: (T \times d_v)$

## 8. Complete Worked Example
Let $T = 2$, $d_{model} = 2$, $d_k = 2$, $d_v = 2$:

$$\mathbf{X} = \begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 0.0 \end{bmatrix}$$

Let projection weights:

$$\mathbf{W}_Q = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix}, \quad \mathbf{W}_K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}, \quad \mathbf{W}_V = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$$

Compute projections:

$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q = \begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 0.0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 1.0 & 4.0 \\ 3.0 & 0.0 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$

$$\mathbf{K} = \mathbf{X} \mathbf{W}_K = \begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 0.0 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 2.0 & 1.0 \\ 0.0 & 3.0 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$

$$\mathbf{V} = \mathbf{X} \mathbf{W}_V = \begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 0.0 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1.0 & 3.0 \\ 3.0 & 3.0 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$

## 9. Math → Code Mapping
```python
import numpy as np

X = np.array([[1.0, 2.0], [3.0, 0.0]]) # Shape (2, 2)

W_Q = np.array([[1, 0], [0, 2]])
W_K = np.array([[0, 1], [1, 0]])
W_V = np.array([[1, 1], [0, 1]])

Q = X @ W_Q # Shape (2, 2)
K = X @ W_K # Shape (2, 2)
V = X @ W_V # Shape (2, 2)

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)
```

## 10. Experiments / What-If Questions
- **What if $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ were identity matrices $\mathbf{I}$?**
  Then $\mathbf{Q} = \mathbf{K} = \mathbf{V} = \mathbf{X}$. The model loses the ability to separate search requests from content profiles.
- **Can $d_k$ differ from $d_{model}$?**
  Yes! $d_k$ and $d_v$ are projection hyper-parameters. In multi-head attention, $d_k = d_{model} / h$.

## 11. Common Misunderstandings
- **Misunderstanding:** Query, Key, and Value are separate input text sequences.
- **Correction:** In Self-Attention, $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ are all projected from the **SAME** input sequence matrix $\mathbf{X}$!

## 12. Limitations and Trade-Offs
Introducing $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ adds $3 \times (d_{model} \cdot d_k)$ trainable parameters to the model layer, requiring gradient tracking for all three projection matrices during backpropagation.

## 13. Where It Appears in the Current Assignment
In **Week 3 Assignment**, you will initialize $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ of shape $(2, 2)$ and compute `Q = X @ W_Q`, `K = X @ W_K`, `V = X @ W_V`.

## 14. Where It Appears in Modern AI Systems
$\mathbf{Q}, \mathbf{K}, \mathbf{V}$ projections form the core input transformations in every Transformer architecture (Vaswani et al., 2017).

## 15. Connection to the Next Concept
Now that we have $\mathbf{Q}$ and $\mathbf{K}$, how do we compute raw score matrix $\mathbf{S} = \mathbf{Q} \mathbf{K}^\top$ and why must we scale it by $\frac{1}{\sqrt{d_k}}$? (`23 - TRANSFORMER - Scaled Dot-Product Attention.md`).

## 16. Teach-Back and Small Application Exercise
If $\mathbf{X}$ has shape $(10, 512)$, $\mathbf{W}_Q$ has shape $(512, 64)$, and $\mathbf{W}_V$ has shape $(512, 64)$:
1. What is the shape of $\mathbf{Q}$?
2. What is the shape of $\mathbf{V}$?

## 17. Quick Revision Summary
- $\mathbf{Q} = \mathbf{X} \mathbf{W}_Q$: What the token is looking for.
- $\mathbf{K} = \mathbf{X} \mathbf{W}_K$: What information the token offers.
- $\mathbf{V} = \mathbf{X} \mathbf{W}_V$: The token's actual content payload.

## 18. My Understanding
*Fill in your own summary of the distinct roles of Query, Key, and Value.*

## 19. Flashcards
What are the roles of Query, Key, and Value in Self-Attention? #card
Query ($Q$) represents what a token is searching for, Key ($K$) represents what information a token contains, and Value ($V$) represents the token's actual content payload.

Are Q, K, and V derived from different text inputs in Self-Attention? #card
No. In Self-Attention, Q, K, and V are all linear projections created from the same single sequence matrix $X$ using weight matrices $W_Q, W_K, W_V$.

## 20. Sources
- Vaswani et al. (2017) *Attention Is All You Need*, Section 3.2.1.
- Alammar, J. *The Illustrated Transformer*.
