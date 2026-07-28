# 20 - MATH - Matrix Transpose and Attention Shapes

## 1. The Problem
Suppose we have two sequence projection matrices, Queries $\mathbf{Q} \in \mathbb{R}^{T \times d_k}$ and Keys $\mathbf{K} \in \mathbb{R}^{T \times d_k}$. 
We want to compare every query row $\mathbf{q}_i$ with every key row $\mathbf{k}_j$ by taking their dot product: $S_{i,j} = \mathbf{q}_i \cdot \mathbf{k}_j$. 
If we try to multiply $\mathbf{Q} \cdot \mathbf{K}$ directly, matrix multiplication is usually undefined because inner dimensions don't match ($d_k \neq T$). Even if $T = d_k$ making the multiplication mathematically legal, $\mathbf{Q} \cdot \mathbf{K}$ does not compute the pairwise row-by-row dot products we actually want.

## 2. Why We Need Something New
To compare every query row with every key row, we need to arrange the keys so that the dot products naturally form a $(T \times T)$ matrix of scores. We need a linear algebra operation that swaps the rows and columns of $\mathbf{K}$, allowing $\mathbf{Q} \mathbf{K}^\top$ to compute exactly what we need.

## 3. One-Line Definition
**Matrix Transpose** (denoted $\mathbf{A}^\top$) flips a matrix over its diagonal, swapping its row and column indices such that an $(M \times N)$ matrix becomes an $(N \times M)$ matrix.

## 4. Beginner Intuition / Mental Model
Imagine a spreadsheet table with 3 rows (tokens) and 2 columns (features). 
Transposing the spreadsheet is simply swapping its rows and columns (or reflecting it across its main diagonal): the 3 rows become 3 columns, and the 2 columns become 2 rows.

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

### Symbol Table

| Symbol | Name | Plain-English Meaning |
| :--- | :--- | :--- |
| $\mathbf{A}^\top$ | **Transpose of Matrix A** | A new matrix created by flipping $\mathbf{A}$ over its diagonal: rows become columns and columns become rows. Shape changes from $(M \times N)$ to $(N \times M)$. |
| $(i, j)$ | **Row-Column Index** | The position of an element at row $i$ and column $j$. After transposing, element at $(i, j)$ moves to $(j, i)$. |
| $\mathbf{Q}$ | **Query Matrix** | The "what am I looking for?" projection of the sequence. Shape: $(T \times d_k)$. (Introduced fully in Topic 22.) |
| $\mathbf{K}$ | **Key Matrix** | The "what information do I contain?" projection. Shape: $(T \times d_k)$. Transposed to $\mathbf{K}^\top$ with shape $(d_k \times T)$ to enable dot-product scoring. |
| $\mathbf{S}$ | **Raw Score Matrix** | The result of $\mathbf{Q} \mathbf{K}^\top$. A square $(T \times T)$ matrix where entry $S_{i,j}$ is the raw dot-product similarity between token $i$'s Query and token $j$'s Key. |
| $d_k$ | **Key/Query Dimension** | The number of features in each Query and Key vector. This is the "inner dimension" that must match for the matrix multiplication to work. |

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
Standard Transformer scaled dot-product attention uses $\mathbf{Q} \mathbf{K}^\top$ matrix transpose dot products to compute attention scores.

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
- Goodfellow, I., Bengio, Y., & Courville, A. [Deep Learning.md](file:///c:/Users/Nagar/source/repos/ai-learning-lab/resources/references/Deep%20Learning.md), Chapter 2 (Linear Algebra).
- Vaswani et al. (2017) *"Attention Is All You Need"*.

