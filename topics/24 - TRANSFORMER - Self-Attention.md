# 24 - TRANSFORMER - Self-Attention

## 1. The Problem
In early seq2seq models (Bahdanau et al., 2014), attention was used *cross-wise* between two distinct networks (an encoder sequence and a decoder sequence). 
However, within a single input sentence itself, words must interact with each other (e.g., resolving pronouns like "it" to "authorization").
**The limitation:** Cross-attention requires two separate sequences; a standalone language model needs a way for tokens in the *same* sequence to attend to each other.

## 2. Why We Need Something New
We need **Self-Attention** (intra-attention), where Query, Key, and Value projections are all generated from the **same** single sequence matrix $\mathbf{X}$.

## 3. One-Line Definition
**Self-Attention** is an attention mechanism where a sequence attends to itself by deriving its Query ($\mathbf{Q}$), Key ($\mathbf{K}$), and Value ($\mathbf{V}$) matrices from the same input sequence $\mathbf{X}$.

## 4. Beginner Intuition / Mental Model
Imagine a **Team Meeting** around a round table:
Every team member (token) speaks up, listens to every other team member in the room, and updates their own understanding based on what everyone else said. The room attends to itself!

## 5. What Came Before → What Changes Now
- **Cross-Attention:** $\mathbf{Q}$ comes from sequence A (decoder); $\mathbf{K}, \mathbf{V}$ come from sequence B (encoder).
- **Self-Attention:** $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ all come from the **same sequence $\mathbf{X}$**!

## 6. How It Works
1. Input sequence matrix $\mathbf{X} \in \mathbb{R}^{T \times d_{model}}$.
2. Project $\mathbf{X}$ into $\mathbf{Q} = \mathbf{X} \mathbf{W}_Q$, $\mathbf{K} = \mathbf{X} \mathbf{W}_K$, $\mathbf{V} = \mathbf{X} \mathbf{W}_V$.
3. Compute self-attention weights $\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{T \times T}$.
4. Compute self-contextual outputs $\mathbf{H} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{T \times d_v}$.

```
Input Tokens ──► X (T x d) ──► Q, K, V Projections ──► Scaled Dot-Product Attention ──► Contextual Outputs H (T x d_v)
```

## 7. Required Mathematics
$$\mathbf{H} = \text{Softmax}\left( \frac{(\mathbf{X} \mathbf{W}_Q) (\mathbf{X} \mathbf{W}_K)^\top}{\sqrt{d_k}} \right) (\mathbf{X} \mathbf{W}_V)$$

**Shape Trace:**
- $\mathbf{X}: (T \times d_{model})$
- $\mathbf{W}_Q, \mathbf{W}_K: (d_{model} \times d_k)$
- $\mathbf{W}_V: (d_{model} \times d_v)$
- $\mathbf{Q}, \mathbf{K}: (T \times d_k)$
- $\mathbf{V}: (T \times d_v)$
- $\mathbf{Q} \mathbf{K}^\top: (T \times T)$
- $\mathbf{A}: (T \times T)$
- $\mathbf{H}: (T \times d_v)$

## 8. Complete Worked Example
Let $T = 3$ tokens (`["Order", "Shipment", "Receive"]`), $d_{model} = 2$, $d_k = 2$, $d_v = 2$.

Input $\mathbf{X} \in \mathbb{R}^{3 \times 2}$.

Projections yield $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{3 \times 2}$.

Dot product $\mathbf{Q} \mathbf{K}^\top$ creates a $3 \times 3$ matrix of scores representing all pairwise relationships:
- Row 0 (`"Order"`): how much `"Order"` attends to `"Order"`, `"Shipment"`, `"Receive"`.
- Row 1 (`"Shipment"`): how much `"Shipment"` attends to `"Order"`, `"Shipment"`, `"Receive"`.
- Row 2 (`"Receive"`): how much `"Receive"` attends to `"Order"`, `"Shipment"`, `"Receive"`.

Softmax normalizes each row into probability weights $\mathbf{A} \in \mathbb{R}^{3 \times 3}$.

Multiplying $\mathbf{A} \mathbf{V}$ produces contextual sequence matrix $\mathbf{H} \in \mathbb{R}^{3 \times 2}$.

## 9. Math → Code Mapping
```python
import numpy as np

class SelfAttentionSingleHead:
    def __init__(self, d_model, d_k, d_v):
        self.W_Q = np.random.randn(d_model, d_k) * 0.1
        self.W_K = np.random.randn(d_model, d_k) * 0.1
        self.W_V = np.random.randn(d_model, d_v) * 0.1
        self.d_k = d_k

    def forward(self, X):
        Q = X @ self.W_Q # (T, d_k)
        K = X @ self.W_K # (T, d_k)
        V = X @ self.W_V # (T, d_v)

        scores = (Q @ K.T) / np.sqrt(self.d_k) # (T, T)
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        A = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

        H = A @ V # (T, d_v)
        return H, A
```

## 10. Experiments / What-If Questions
- **What happens if two tokens in the sequence are identical copies?**
  They will produce identical Query and Key vectors, receiving identical attention scores from all other tokens.

## 11. Common Misunderstandings
- **Misunderstanding:** Self-attention requires recurrent step-by-step loops over time like an RNN.
- **Correction:** Self-attention computes all pairwise token interactions in **parallel** in a single matrix multiplication $\mathbf{Q} \mathbf{K}^\top$!

## 12. Limitations and Trade-Offs
- **Permutation Equivariance:** Self-attention is order-blind! If you shuffle the input rows of $\mathbf{X}$, the output rows of $\mathbf{H}$ shuffle identically. (This is why Positional Encoding is required in Week 4!).

## 13. Where It Appears in the Current Assignment
In **Week 3 Assignment**, you will build a standalone NumPy `SelfAttentionSingleHead` class and test it on 3 operational log tokens.

## 14. Where It Appears in Modern AI Systems
Self-Attention is the core building block of all Transformer Encoders (BERT) and Decoders (GPT, Llama, Claude).

## 15. Connection to the Next Concept
In language modeling (predicting the next token), allowing token $t$ to attend to future tokens $> t$ is **cheating**! How do we block peeking into the future? `25 - TRANSFORMER - Causal Masking.md`.

## 16. Teach-Back and Small Application Exercise
1. In Self-Attention, where do $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ come from?
2. What is the shape of the attention matrix $\mathbf{A}$ for a sequence of $T = 5$ tokens?

## 17. Quick Revision Summary
- Self-Attention derives $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ from the same input sequence $\mathbf{X}$.
- Operates in parallel across all token pairs ($T \times T$).
- Order-blind without positional encoding.

## 18. My Understanding
*Fill in your own summary of how Self-Attention processes sequence interaction.*

## 19. Flashcards
What distinguishes Self-Attention from Cross-Attention? #card
In Self-Attention, Query, Key, and Value matrices are all projected from the same single input sequence. In Cross-Attention, Query comes from one sequence while Key and Value come from another sequence.

Why is Self-Attention order-blind (permutation-equivariant) by default? #card
Because matrix dot products $Q K^\top$ compute set-wise similarities between vectors regardless of their row positions in the sequence matrix.

## 20. Sources
- Vaswani et al. (2017) *Attention Is All You Need*, Section 3.2.
- Alammar, J. *The Illustrated Transformer*.
