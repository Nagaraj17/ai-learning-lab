# 21 - TRANSFORMER - Attention and Contextual Representations

## 1. The Problem
Consider the word **`bank`** in two different sentences:
- **Sentence 1:** *"I sat by the river bank."*
- **Sentence 2:** *"I deposited money at the bank."*

In Week 2, our model used a static Embedding Matrix $\mathbf{E}$. Word `bank` has one fixed row in $\mathbf{E}$. Therefore, in both sentences, `bank` receives the **EXACT SAME** vector $\mathbf{v}_{\text{bank}} = [0.45, -0.12]$.
**The limitation:** Static embeddings cannot alter their vector values based on the surrounding sentence context.

## 2. Why We Need Something New
We need a dynamic mechanism that allows token vectors to interact with neighboring tokens in the sequence, pulling relevant context from surrounding words to update their representations at runtime.

## 3. One-Line Definition
**Attention** is a mechanism that computes dynamic, context-dependent representations ($\mathbf{H}$) by taking a weighted sum of sequence token representations based on pairwise interaction scores.

## 4. Beginner Intuition / Mental Model
Imagine static embeddings as **dictionary definitions**. A dictionary entry for "bank" lists all meanings at once.
Attention is like a **smart translator**: when reading a sentence, it looks at the surrounding words ("river" or "money") and highlights only the relevant meaning, creating a customized, context-aware profile for "bank" in that specific sentence.

## 5. What Came Before → What Changes Now
- **Before (Bahdanau et al., 2014):** Attention was introduced in seq2seq RNNs to allow a decoder to focus on specific source encoder states instead of compressing an entire sentence into one bottleneck vector.
- **Now (Vaswani et al., 2017):** **Self-Attention** eliminates RNNs entirely, allowing every token in a sequence to attend directly to every other token in parallel!

## 6. How It Works
1. Start with static sequence matrix $\mathbf{X} \in \mathbb{R}^{T \times d_{model}}$.
2. Measure pairwise relevance between every token pair $(i, j)$ in the sequence.
3. Convert relevance scores into normalized attention probabilities (weights that sum to $1.0$).
4. Mix the token representations according to those attention weights to produce contextual outputs $\mathbf{H} \in \mathbb{R}^{T \times d_v}$.

```
Static Embeddings X (T x d)  ──► [ Attention Mechanism ] ──► Contextual Representations H (T x d)
  "bank" = [0.45, -0.12]                                      "bank" (Sentence 1) = [0.05, 0.92] (Nature)
  "bank" = [0.45, -0.12]                                      "bank" (Sentence 2) = [0.95, -0.80] (Finance)
```

## 7. Required Mathematics
Let $\mathbf{A} \in \mathbb{R}^{T \times T}$ be the attention weight matrix where row $i$ gives the probability distribution over all tokens for token $i$ ($\sum_{j=1}^T A_{i, j} = 1.0$).

Let $\mathbf{V} \in \mathbb{R}^{T \times d_v}$ be the representation matrix of the sequence.

The Contextual Output Matrix $\mathbf{H}$ is computed as:

$$\mathbf{H} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{T \times d_v}$$

**Shape Trace:**
- Attention Weights $\mathbf{A}$: $(T \times T)$
- Values $\mathbf{V}$: $(T \times d_v)$
- Contextual Output $\mathbf{H}$: $(T \times d_v)$

### Symbol Table

| Symbol | Name | Plain-English Meaning |
| :--- | :--- | :--- |
| $\mathbf{A}$ | **Attention Weight Matrix** | A $(T \times T)$ matrix where entry $A_{i,j}$ is the probability (0.0 to 1.0) of how much token $i$ should attend to token $j$. Each row sums to $1.0$. |
| $A_{i,j}$ | **Attention Weight Entry** | A single scalar between $0.0$ and $1.0$ representing: "How much should token $i$ pull information from token $j$?" |
| $\mathbf{V}$ | **Value Matrix** | A $(T \times d_v)$ matrix containing the "content payload" vectors for each token. This is what gets mixed together by attention. |
| $d_v$ | **Value Dimension** | The number of features in each Value vector. Often $d_v = d_k$, but they can differ. |
| $\mathbf{H}$ | **Contextual Output Matrix** | The $(T \times d_v)$ result of $\mathbf{A} \mathbf{V}$. Each row $i$ is a new vector for token $i$ that blends information from all tokens weighted by attention. |
| $\sum_{j=1}^T A_{i,j} = 1.0$ | **Row Sum Constraint** | Each row of $\mathbf{A}$ is a valid probability distribution — the attention weights for any given token must sum to exactly $1.0$ (guaranteed by Softmax). |

## 8. Complete Worked Example
Let sequence length $T = 2$ (`["river", "bank"]`), and $d_v = 2$.

Let Values $\mathbf{V} = \begin{bmatrix} 0.1 & 0.9 \\ 0.4 & 0.2 \end{bmatrix}$ where Row 0 is `"river"` and Row 1 is `"bank"`.

Suppose attention weights $\mathbf{A} = \begin{bmatrix} 0.9 & 0.1 \\ 0.7 & 0.3 \end{bmatrix}$:
- Token 0 (`"river"`) pays 90% attention to itself and 10% to `"bank"`.
- Token 1 (`"bank"`) pays 70% attention to `"river"` and 30% to itself.

Compute contextual output $\mathbf{H} = \mathbf{A} \mathbf{V}$:

- **Row 0 (`"river"` output):**
  $$\mathbf{H}_{row 0} = 0.9 \times [0.1, 0.9] + 0.1 \times [0.4, 0.2] = [0.09, 0.81] + [0.04, 0.02] = [0.13, 0.83]$$
  `"river"` stays mostly like itself (90% self-attention), with a tiny pull from `"bank"`.

- **Row 1 (`"bank"` output):**
  $$\mathbf{H}_{row 1} = 0.7 \times [0.1, 0.9] + 0.3 \times [0.4, 0.2] = [0.07, 0.63] + [0.12, 0.06] = [0.19, 0.69]$$
  `"bank"`'s output vector $[0.19, 0.69]$ is now heavily influenced by `"river"`!

$$\mathbf{H} = \begin{bmatrix} 0.13 & 0.83 \\ 0.19 & 0.69 \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$

> **Key Observation:** Both tokens got updated — not just `"bank"`. Every row of $\mathbf{H}$ is a weighted blend of ALL Value rows. The attention weights determine the blend ratios.

## 9. Math → Code Mapping
```python
import numpy as np

# A: Attention weights (2, 2)
A = np.array([
    [0.9, 0.1],
    [0.7, 0.3]
])

# V: Value representations (2, 2)
V = np.array([
    [0.1, 0.9], # river
    [0.4, 0.2]  # bank
])

# Contextual output H = A @ V
H = A @ V # Shape (2, 2)
print("Contextual representation for 'bank':", H[1])
```

## 10. Experiments / What-If Questions
- **What if all attention weights in a row are equal ($\frac{1}{T}$)?**
  Attention degenerates into a uniform average (Bag-of-Words), treating every word as equally relevant.
- **What if an attention weight is $1.0$ for self and $0.0$ for others?**
  The token receives zero context from surrounding words, preserving its un-contextualized representation.

## 11. Common Misunderstandings
- **Misunderstanding:** Attention weights can be automatically interpreted as complete explanations of a model's reasoning.
- **Correction (Jain & Wallace, 2019):** Attention weights show correlation in feature mixing, but *attention is not explanation*. Alternative attention distributions can yield identical final model outputs.
- **Misunderstanding:** An untrained or small model will automatically produce intuitive, human-like attention weights.
- **Correction:** Attention weights are driven by gradient descent optimization; untrained models produce random attention weights.

## 12. Limitations and Trade-Offs
Computing pairwise attention weights requires an $O(T^2)$ matrix $\mathbf{A} \in \mathbb{R}^{T \times T}$. For long sequences ($T = 100,000$), storing and computing $T^2$ attention scores becomes computationally expensive.

## 13. Where It Appears in the Current Assignment
In **Week 3 Assignment**, you will verify how attention matrix $\mathbf{A} \in \mathbb{R}^{3 \times 3}$ transforms static log embeddings into contextual representations $\mathbf{H} \in \mathbb{R}^{3 \times 2}$.

## 14. Where It Appears in Modern AI Systems
Every modern Transformer block uses attention to convert static input embeddings into deep contextual representations across multiple layers.

## 15. Connection to the Next Concept
How does the model decide *which* tokens should attend to which? It creates separate roles using **Query, Key, and Value** projections (`22 - TRANSFORMER - Query Key and Value.md`).

## 16. Teach-Back and Small Application Exercise
If $\mathbf{A}$ has shape $(4, 4)$ and $\mathbf{V}$ has shape $(4, 8)$:
1. What is the shape of contextual output $\mathbf{H} = \mathbf{A} \mathbf{V}$?
2. What must the sum of elements in each row of $\mathbf{A}$ equal?

## 17. Quick Revision Summary
- Static embeddings give frozen vectors regardless of sentence context.
- Attention computes dynamic contextual representations $\mathbf{H} = \mathbf{A} \mathbf{V}$.
- Attention weights $\mathbf{A}$ determine how much context is pulled from each word.

## 18. My Understanding
*Fill in your own notes on how Attention turns static embeddings into contextual representations.*

## 19. Flashcards
What is the core difference between a static embedding and a contextual representation? #card
A static embedding provides the same frozen vector for a token regardless of context. A contextual representation dynamically adjusts a token's vector based on surrounding words in the sequence.

Can attention weights always be treated as definitive explanations of model decision making? #card
No (Jain & Wallace, 2019). Attention weights show feature mixing weights, but different attention distributions can yield identical predictions. They are not guaranteed explanation proofs.

## 20. Sources
- Bahdanau, Cho, & Bengio (2014) *"Neural Machine Translation by Jointly Learning to Align and Translate"*.
- Vaswani et al. (2017) *"Attention Is All You Need"*.
- Jain & Wallace (2019) *"Attention is not Explanation"*.
- Alammar, J. & Grootendorst, M. [Hands-On Large Language Models.md](file:///c:/Users/Nagar/source/repos/ai-learning-lab/resources/references/Hands-On%20Large%20Language%20Models.md), Chapter 3.

