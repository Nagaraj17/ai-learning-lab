# 20 - MATH - Matrix Transpose and Attention Shapes

## 1. The Problem
Suppose we have a sequence matrix $\mathbf{X} \in \mathbb{R}^{T \times d}$. 
We want to compare every token against every other token by taking matrix dot products. 
If we try to multiply $\mathbf{X} \cdot \mathbf{X}$, matrix multiplication **FAILS** because the inner dimensions do not match: $(T \times d) \cdot (T \times d)$ is mathematically undefined ($d \neq T$).

## 2. Why We Need Something New
We need a linear algebra operation that flips the rows and columns of a matrix so that inner dimensions align for pairwise token dot products: $(T \times d) \cdot (d \times T) = (T \times T)$.

## 3. One-Line Definition
**Matrix Transpose** (denoted $\mathbf{A}^\top$) flips a matrix over its diagonal, swapping its row and column indices such that an $(M \times N)$ matrix becomes an $(N \times M)$ matrix.

## 4. Beginner Intuition / Mental Model
Imagine a spreadsheet table with 3 rows (tokens) and 2 columns (features). 
Transposing the spreadsheet is like rotating the table 90 degrees and flipping it: the 3 rows become 3 columns, and the 2 columns become 2 rows.

## 5. What Came Before → What Changes Now
- **Before:** Standard matrix multiplication $\mathbf{A} \mathbf{B}$ where $A_{cols} == B_{rows}$.
- **Now:** Transposing $\mathbf{K} \in \mathbb{R}^{T \times d_k} \implies \mathbf{K}^\top \in \mathbb{R}^{d_k \times T}$, allowing $\mathbf{Q} \mathbf{K}^\top$ to produce a square pairwise similarity matrix $(T \times T)$.

## 6. How It Works
For any entry at row $i$, column $j$ in matrix $\mathbf{A}$:

$$(\mathbf{A}^\top)_{j, i} = \mathbf{A}_{i, j}$$

1. Row 0 of $\mathbf{A}$ becomes Column 0 of $\mathbf{A}^\top$.
2. Row 1 of $\mathbf{A}$ becomes Column 1 of $\mathbf{A}^\top$.

## 7. Required Mathematics
For a matrix $\mathbf{K} \in \mathbb{R}^{T \times d_k}$:

$$\mathbf{K} = \begin{bmatrix} k_{1,1} & k_{1,2} \\ k_{2,1} & k_{2,2} \\ k_{3,1} & k_{3,2} \end{bmatrix} \in \mathbb{R}^{3 \times 2} \implies \mathbf{K}^\top = \begin{bmatrix} k_{1,1} & k_{2,1} & k_{3,1} \\ k_{1,2} & k_{2,2} & k_{3,2} \end{bmatrix} \in \mathbb{R}^{2 \times 3}$$

**Shape Trace for Attention Scores:**
$$\mathbf{Q} \in \mathbb{R}^{T \times d_k}, \quad \mathbf{K}^\top \in \mathbb{R}^{d_k \times T}$$
$$\mathbf{S} = \mathbf{Q} \cdot \mathbf{K}^\top \in \mathbb{R}^{(T \times d_k) \cdot (d_k \times T)} = \mathbb{R}^{T \times T}$$

## 8. Complete Worked Example
Let $\mathbf{K} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$.

Transposing gives $\mathbf{K}^\top = \begin{bmatrix} 1 & 3 & 5 \\ 2 & 4 & 6 \end{bmatrix} \in \mathbb{R}^{2 \times 3}$.

Let $\mathbf{Q} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$.

Compute $\mathbf{S} = \mathbf{Q} \mathbf{K}^\top$:

$$\mathbf{S} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 3 & 5 \\ 2 & 4 & 6 \end{bmatrix} = \begin{bmatrix} (1\cdot1 + 0\cdot2) & (1\cdot3 + 0\cdot4) & (1\cdot5 + 0\cdot6) \\ (0\cdot1 + 1\cdot2) & (0\cdot3 + 1\cdot4) & (0\cdot5 + 1\cdot6) \\ (1\cdot1 + 1\cdot2) & (1\cdot3 + 1\cdot4) & (1\cdot5 + 1\cdot6) \end{bmatrix} = \begin{bmatrix} 1 & 3 & 5 \\ 2 & 4 & 6 \\ 3 & 7 & 11 \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$

The result is a square $3 \times 3$ grid of pairwise dot products!

## 9. Math → Code Mapping
```python
import numpy as np

K = np.array([[1, 2], [3, 4], [5, 6]]) # Shape (3, 2)
K_T = K.T                             # Transpose -> Shape (2, 3)

Q = np.array([[1, 0], [0, 1], [1, 1]]) # Shape (3, 2)
S = Q @ K_T                            # Shape (3, 2) @ (2, 3) -> (3, 3)

print("K_T shape:", K_T.shape)
print("S shape:", S.shape)
```

## 10. Experiments / What-If Questions
- **What happens if you transpose a matrix twice $(\mathbf{A}^\top)^\top$?**
  You return to the original matrix $\mathbf{A}$.
- **What happens if $\mathbf{Q}$ and $\mathbf{K}$ have different feature dimensions ($d_q \neq d_k$)?**
  $\mathbf{Q} \mathbf{K}^\top$ will fail inner dimension alignment. $\mathbf{Q}$ and $\mathbf{K}$ MUST project to the exact same hidden size $d_k$.

## 11. Common Misunderstandings
- **Misunderstanding:** Transposing a matrix inverts its numerical values or negates them.
- **Correction:** Transposing only changes element positions $(i, j) \to (j, i)$; it does not alter numerical values.

## 12. Limitations and Trade-Offs
Transposing a matrix in memory requires reshaping or stride manipulation. In PyTorch/NumPy, `K.T` or `K.transpose(-2, -1)` creates a view, but non-contiguous memory layouts may require `.contiguous()` before certain ops.

## 13. Where It Appears in the Current Assignment
In **Week 3 Assignment**, you will calculate `K.T` to compute the raw score matrix `S = Q @ K.T` of shape `(3, 3)`.

## 14. Where It Appears in Modern AI Systems
Every Attention mechanism in Transformers relies on $\mathbf{Q} \mathbf{K}^\top$ matrix transpose dot products.

## 15. Connection to the Next Concept
Now that we can compute square score matrices $(T \times T)$, why do static embeddings need attention to become dynamic contextual representations? (`21 - TRANSFORMER - Attention and Contextual Representations.md`).

## 16. Teach-Back and Small Application Exercise
If $\mathbf{K}$ has shape $(128, 64)$:
1. What is the shape of $\mathbf{K}^\top$?
2. If $\mathbf{Q}$ has shape $(128, 64)$, what is the shape of $\mathbf{Q} \mathbf{K}^\top$?

## 17. Quick Revision Summary
- Transpose flips rows and columns: $(M \times N) \implies (N \times M)$.
- $\mathbf{Q} \mathbf{K}^\top$ uses transpose to align inner dimension $d_k$, producing pairwise matrix of shape $(T \times T)$.

## 18. My Understanding
*Fill in your own notes on how matrix transpose enables pairwise token comparisons.*

## 19. Flashcards
If matrix $\mathbf{A}$ has shape $(T, d_k)$, what is the shape of its transpose $\mathbf{A}^\top$? #card
The shape of $\mathbf{A}^\top$ is $(d_k, T)$.

Why is matrix transpose necessary in Attention score computation $\mathbf{Q} \mathbf{K}^\top$? #card
Because $\mathbf{Q}$ and $\mathbf{K}$ both have shape $(T, d_k)$. Transposing $\mathbf{K}$ to $(d_k, T)$ aligns the inner dimensions for matrix multiplication, producing a square pairwise score matrix of shape $(T, T)$.

## 20. Sources
- Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning*, Chapter 2 (Linear Algebra).
- Vaswani et al. (2017) *Attention Is All You Need*.
