# MANUAL EXERCISE: Single-Head Causal Self-Attention

> **Objective:** Perform end-to-end hand calculation of a Single-Head Causal Self-Attention layer for a sequence of 3 tokens with 2-dimensional embeddings, verifying every intermediate matrix shape and value.

---

## Given Data

Sequence length $T = 3$, $d_{model} = 2$, $d_k = 2$, $d_v = 2$.
Scaling factor $\sqrt{d_k} = \sqrt{2} \approx 1.414$.

### Input Sequence Matrix $\mathbf{X} \in \mathbb{R}^{3 \times 2}$
$$\mathbf{X} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \\ 1.0 & 1.0 \end{bmatrix}$$

### Linear Projection Weights
$$\mathbf{W}_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad \mathbf{W}_K = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad \mathbf{W}_V = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$$

---

## Learner Workspace

### Step 1: Compute $\mathbf{Q} = \mathbf{X} \mathbf{W}_Q$
$$\mathbf{Q} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \\ 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} \quad & \quad \\ \quad & \quad \\ \quad & \quad \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

### Step 2: Compute $\mathbf{K} = \mathbf{X} \mathbf{W}_K$
$$\mathbf{K} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \\ 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} \quad & \quad \\ \quad & \quad \\ \quad & \quad \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

### Step 3: Compute $\mathbf{V} = \mathbf{X} \mathbf{W}_V$
$$\mathbf{V} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \\ 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} \quad & \quad \\ \quad & \quad \\ \quad & \quad \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

### Step 4: Transpose Keys $\mathbf{K}^\top$
$$\mathbf{K}^\top = \begin{bmatrix} \quad & \quad & \quad \\ \quad & \quad & \quad \end{bmatrix} \in \mathbb{R}^{2 \times 3}$$

### Step 5: Compute Raw Scores $\mathbf{S} = \mathbf{Q} \mathbf{K}^\top$
$$\mathbf{S} = \mathbf{Q} \mathbf{K}^\top = \begin{bmatrix} \quad & \quad & \quad \\ \quad & \quad & \quad \\ \quad & \quad & \quad \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$

### Step 6: Scale Scores $\mathbf{S}_{\text{scaled}} = \mathbf{S} / 1.414$
$$\mathbf{S}_{\text{scaled}} = \begin{bmatrix} \quad & \quad & \quad \\ \quad & \quad & \quad \\ \quad & \quad & \quad \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$

### Step 7: Apply Causal Mask (Add $-\infty$ to upper triangle $j > i$)
$$\mathbf{S}_{\text{masked}} = \begin{bmatrix} \quad & -\infty & -\infty \\ \quad & \quad & -\infty \\ \quad & \quad & \quad \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$

### Step 8: Row-wise Softmax $\mathbf{A} = \text{Softmax}(\mathbf{S}_{\text{masked}})$
$$\mathbf{A} = \begin{bmatrix} \quad & 0.000 & 0.000 \\ \quad & \quad & 0.000 \\ \quad & \quad & \quad \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$

### Step 9: Compute Contextual Output $\mathbf{H} = \mathbf{A} \mathbf{V}$
$$\mathbf{H} = \mathbf{A} \mathbf{V} = \begin{bmatrix} \quad & \quad \\ \quad & \quad \\ \quad & \quad \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

---

## Solution Key (For Self-Verification)

1. $\mathbf{Q} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \\ 1.0 & 1.0 \end{bmatrix}$, $\mathbf{K} = \begin{bmatrix} 1.0 & 1.0 \\ 0.0 & 2.0 \\ 1.0 & 2.0 \end{bmatrix}$, $\mathbf{V} = \begin{bmatrix} 2.0 & 0.0 \\ 0.0 & 2.0 \\ 2.0 & 1.0 \end{bmatrix}$
2. $\mathbf{K}^\top = \begin{bmatrix} 1.0 & 0.0 & 1.0 \\ 1.0 & 2.0 & 2.0 \end{bmatrix}$
3. $\mathbf{S} = \begin{bmatrix} 1.0 & 0.0 & 1.0 \\ 2.0 & 4.0 & 4.0 \\ 2.0 & 2.0 & 3.0 \end{bmatrix}$
4. $\mathbf{S}_{\text{scaled}} \approx \begin{bmatrix} 0.707 & 0.000 & 0.707 \\ 1.414 & 2.828 & 2.828 \\ 1.414 & 1.414 & 2.121 \end{bmatrix}$
5. $\mathbf{S}_{\text{masked}} = \begin{bmatrix} 0.707 & -\infty & -\infty \\ 1.414 & 2.828 & -\infty \\ 1.414 & 1.414 & 2.121 \end{bmatrix}$
6. $\mathbf{A} \approx \begin{bmatrix} 1.000 & 0.000 & 0.000 \\ 0.196 & 0.804 & 0.000 \\ 0.248 & 0.248 & 0.504 \end{bmatrix}$
7. $\mathbf{H} \approx \begin{bmatrix} 2.000 & 0.000 \\ 0.392 & 1.608 \\ 1.504 & 1.000 \end{bmatrix}$
