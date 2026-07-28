# PREREQUISITE KNOWLEDGE (Week 3: Attention & Contextual Representations)

This is the consolidated study guide for **Week 3 (Attention & Contextual Representations)**. It covers every new concept, mathematical operation, and shape transformation required for the Week 3 assignment.

---

## 1. Sequence Representations & Matrix Shapes
In Week 1 and Week 2, our neural network processed single isolated tokens $\mathbf{v} \in \mathbb{R}^{d_{model}}$.
In language modeling, we process context windows of length $T$. Stacking $T$ token embedding vectors along rows produces the **Sequence Matrix** $\mathbf{X}$:

$$\mathbf{X} = \begin{bmatrix} \mathbf{e}_{t_1} \\ \mathbf{e}_{t_2} \\ \vdots \\ \mathbf{e}_{t_T} \end{bmatrix} \in \mathbb{R}^{T \times d_{model}}$$

**Shape Trace:**
- Token IDs: $(T,)$
- Embedding Matrix $\mathbf{E}$: $(|V| \times d_{model})$
- Sequence Matrix $\mathbf{X}$: $(T \times d_{model})$

---

## 2. The Limitations of Static Embeddings
Static embeddings assign one frozen vector per word from the lookup table $\mathbf{E}$.
- Word `"bank"` in *"river bank"* $\to [0.45, -0.12]$
- Word `"bank"` in *"money bank"* $\to [0.45, -0.12]$

Static embeddings cannot adapt to surrounding context. We need **Attention** to take a weighted sum of sequence token representations and produce dynamic **Contextual Representations** $\mathbf{H} \in \mathbb{R}^{T \times d_v}$.

---

## 3. Query, Key, and Value (Q, K, V) Projections
Instead of comparing raw sequence vectors $\mathbf{X}$ directly, we project $\mathbf{X}$ into three separate functional roles using linear weight matrices $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{d_{model} \times d_k}$ and $\mathbf{W}_V \in \mathbb{R}^{d_{model} \times d_v}$:

- **Query ($\mathbf{Q}$):** What a token is searching for $\mathbf{Q} = \mathbf{X} \mathbf{W}_Q \in \mathbb{R}^{T \times d_k}$
- **Key ($\mathbf{K}$):** What information a token contains $\mathbf{K} = \mathbf{X} \mathbf{W}_K \in \mathbb{R}^{T \times d_k}$
- **Value ($\mathbf{V}$):** A token's actual content payload $\mathbf{V} = \mathbf{X} \mathbf{W}_V \in \mathbb{R}^{T \times d_v}$

---

## 4. Scaled Dot-Product Attention Math & Shapes
To measure pairwise token interactions, we compute raw dot products between queries and keys:

$$\mathbf{S} = \mathbf{Q} \mathbf{K}^\top \in \mathbb{R}^{T \times T}$$

### Why scale by $\frac{1}{\sqrt{d_k}}$?
For large key dimension $d_k$, variance of dot products increases ($\text{Var} = d_k$). Large unscaled score values push Softmax into extreme saturation regions where gradients vanish ($\text{Softmax}'(z) \to 0$). Dividing by $\sqrt{d_k}$ normalizes variance back to $1.0$.

$$\mathbf{S}_{\text{scaled}} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}$$

Row-wise Softmax converts raw scores into normalized attention probability weights $\mathbf{A} \in \mathbb{R}^{T \times T}$ where row $i$ sums to $1.0$.

---

## 5. Causal Masking (Autoregressive Property)
In autoregressive next-token prediction, token $i$ must not look into future positions $j > i$.
We add a lower-triangular Causal Mask Matrix $\mathbf{M} \in \mathbb{R}^{T \times T}$ to the scaled scores before Softmax:

$$M_{i, j} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$

$$\mathbf{A} = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M} \right) \in \mathbb{R}^{T \times T}$$

Because $e^{-\infty} = 0.0$, future positions receive **exactly 0% attention**.

---

## 6. Single-Head Self-Attention Output
The contextual representation matrix $\mathbf{H}$ is computed as the weighted sum of Value representations:

$$\mathbf{H} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{T \times d_v}$$

**Complete Central Shape Trace:**
$$(T \times d_{model}) \xrightarrow{\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V} \mathbf{Q}, \mathbf{K}, \mathbf{V}: (T \times d_k), (T \times d_v) \xrightarrow{\mathbf{Q} \mathbf{K}^\top} \mathbf{S}: (T \times T) \xrightarrow{\text{Mask+Softmax}} \mathbf{A}: (T \times T) \xrightarrow{\mathbf{A} \mathbf{V}} \mathbf{H}: (T \times d_v)$$

---

## 7. Complete Hand-Calculated Worked Numerical Example

Let sequence length $T = 3$ tokens (`["Order", "Shipment", "Receive"]`), hidden size $d_{model} = 2$, key dimension $d_k = 2$, value dimension $d_v = 2$.

$$\text{Scaling factor } \sqrt{d_k} = \sqrt{2} \approx 1.414$$

### Step 1: Input Sequence Matrix $\mathbf{X} \in \mathbb{R}^{3 \times 2}$
$$\mathbf{X} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \\ 1.0 & 1.0 \end{bmatrix}$$

### Step 2: Linear Projection Weights
$$\mathbf{W}_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad \mathbf{W}_K = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad \mathbf{W}_V = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}$$

### Step 3: Compute Projections $\mathbf{Q}, \mathbf{K}, \mathbf{V}$
$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \\ 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \\ 1.0 & 1.0 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

$$\mathbf{K} = \mathbf{X} \mathbf{W}_K = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \\ 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1.0 & 1.0 \\ 0.0 & 2.0 \\ 1.0 & 2.0 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

$$\mathbf{V} = \mathbf{X} \mathbf{W}_V = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \\ 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 2.0 & 0.0 \\ 0.0 & 2.0 \\ 2.0 & 1.0 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

### Step 4: Transpose Keys $\mathbf{K}^\top \in \mathbb{R}^{2 \times 3}$
$$\mathbf{K}^\top = \begin{bmatrix} 1.0 & 0.0 & 1.0 \\ 1.0 & 2.0 & 2.0 \end{bmatrix}$$

### Step 5: Compute Raw Scores $\mathbf{S} = \mathbf{Q} \mathbf{K}^\top \in \mathbb{R}^{3 \times 3}$
$$\mathbf{S} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \\ 1.0 & 1.0 \end{bmatrix} \begin{bmatrix} 1.0 & 0.0 & 1.0 \\ 1.0 & 2.0 & 2.0 \end{bmatrix} = \begin{bmatrix} 1.0 & 0.0 & 1.0 \\ 2.0 & 4.0 & 4.0 \\ 2.0 & 2.0 & 3.0 \end{bmatrix}$$

### Step 6: Scale Scores by $\sqrt{2} \approx 1.414$
$$\mathbf{S}_{\text{scaled}} = \frac{\mathbf{S}}{1.414} \approx \begin{bmatrix} 0.707 & 0.000 & 0.707 \\ 1.414 & 2.828 & 2.828 \\ 1.414 & 1.414 & 2.121 \end{bmatrix}$$

### Step 7: Apply Causal Mask $\mathbf{M}$ (Set upper triangle $j > i$ to $-\infty$)
$$\mathbf{S}_{\text{masked}} = \begin{bmatrix} 0.707 & -\infty & -\infty \\ 1.414 & 2.828 & -\infty \\ 1.414 & 1.414 & 2.121 \end{bmatrix}$$

### Step 8: Row-wise Softmax $\mathbf{A} = \text{Softmax}(\mathbf{S}_{\text{masked}})$
- **Row 0 (`"Order"`):** $\text{Softmax}([0.707, -\infty, -\infty]) = [1.000, 0.000, 0.000]$
- **Row 1 (`"Shipment"`):** $\text{Softmax}([1.414, 2.828, -\infty])$
  - $e^{1.414} \approx 4.112$, $e^{2.828} \approx 16.911$, Sum $\approx 21.023$
  - $\mathbf{A}_{\text{row 1}} = [4.112 / 21.023, 16.911 / 21.023, 0.000] \approx [0.196, 0.804, 0.000]$
- **Row 2 (`"Receive"`):** $\text{Softmax}([1.414, 1.414, 2.121])$
  - $e^{1.414} \approx 4.112$, $e^{1.414} \approx 4.112$, $e^{2.121} \approx 8.339$, Sum $\approx 16.563$
  - $\mathbf{A}_{\text{row 2}} = [4.112 / 16.563, 4.112 / 16.563, 8.339 / 16.563] \approx [0.248, 0.248, 0.504]$

$$\mathbf{A} = \begin{bmatrix} 1.000 & 0.000 & 0.000 \\ 0.196 & 0.804 & 0.000 \\ 0.248 & 0.248 & 0.504 \end{bmatrix} \in \mathbb{R}^{3 \times 3}$$

### Step 9: Compute Contextual Output $\mathbf{H} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{3 \times 2}$
$$\mathbf{H} = \begin{bmatrix} 1.000 & 0.000 & 0.000 \\ 0.196 & 0.804 & 0.000 \\ 0.248 & 0.248 & 0.504 \end{bmatrix} \begin{bmatrix} 2.0 & 0.0 \\ 0.0 & 2.0 \\ 2.0 & 1.0 \end{bmatrix}$$

- **Row 0 (`"Order"` output):**
  $$1.000 \times [2.0, 0.0] + 0 = [2.000, 0.000]$$
- **Row 1 (`"Shipment"` output):**
  $$0.196 \times [2.0, 0.0] + 0.804 \times [0.0, 2.0] = [0.392, 1.608]$$
- **Row 2 (`"Receive"` output):**
  $$0.248 \times [2.0, 0.0] + 0.248 \times [0.0, 2.0] + 0.504 \times [2.0, 1.0] = [0.496 + 1.008, 0.496 + 0.504] = [1.504, 1.000]$$

$$\mathbf{H} = \begin{bmatrix} 2.000 & 0.000 \\ 0.392 & 1.608 \\ 1.504 & 1.000 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

Every calculation step matches matrix algebra exactly!

---

## 8. Manual Verification Exercises
Complete the hand calculations in:
- [Level 3: Single-Head Self-Attention](manual-exercises/01_SINGLE_HEAD_ATTENTION.md)

---

## 9. References & Sources
- Vaswani et al. (2017) *"Attention Is All You Need"*
- Bahdanau, Cho, & Bengio (2014) *"Neural Machine Translation by Jointly Learning to Align and Translate"*
- Jain & Wallace (2019) *"Attention is not Explanation"*
- Alammar, J. & Grootendorst, M. [Hands-On Large Language Models.md](file:///c:/Users/Nagar/source/repos/ai-learning-lab/resources/references/Hands-On%20Large%20Language%20Models.md)
- Goodfellow, I., Bengio, Y., & Courville, A. [Deep Learning.md](file:///c:/Users/Nagar/source/repos/ai-learning-lab/resources/references/Deep%20Learning.md)
- Raschka, S. [Build a Large Language Model (From Scratch).md](file:///c:/Users/Nagar/source/repos/ai-learning-lab/resources/references/Build%20a%20Large%20Language%20Model%20(From%20Scratch).md)

