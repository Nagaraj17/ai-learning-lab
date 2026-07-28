# 19 - LM - Context Windows and Sequence Representations

## 1. The Problem
In Week 1 and Week 2, our neural network processed **one single token at a time** (e.g. `Input: "Receive" -> Predict: "Restock"`).
However, human language and operational logs do not exist in isolated single-word chunks. Consider sentence A vs sentence B:
- **Sentence A:** `"Order shipped today."`
- **Sentence B:** `"Order canceled today."`

If our model only looks at the single word `"today"`, it has zero idea whether the next token should be `"Receive"` or `"Alert"`. 
**The limitation:** Processing tokens as isolated 1D vectors destroys sentence history and context.

## 2. Why We Need Something New
We need a mathematical structure that can hold an **entire sequence of $T$ tokens** simultaneously in memory, preserving the order and feature vectors of every token in the context window.

## 3. One-Line Definition
A **Sequence Matrix** $\mathbf{X} \in \mathbb{R}^{T \times d_{model}}$ is a 2D matrix formed by stacking the $d_{model}$-dimensional embedding vectors of $T$ sequential tokens along the rows.

## 4. Beginner Intuition / Mental Model
Imagine a single token embedding as a **single page profile** of a person. 
A Sequence Matrix is a **bound book of $T$ pages**, where Page 1 is Token 1, Page 2 is Token 2, and Page $T$ is Token $T$. The book holds the entire story of the context window.

## 5. What Came Before → What Changes Now
- **Before (Week 2):** Single token vector $\mathbf{v} \in \mathbb{R}^{d_{model}}$ (Shape: $1 \times d_{model}$).
- **Now (Week 3):** Context Window Sequence Matrix $\mathbf{X} \in \mathbb{R}^{T \times d_{model}}$ (Shape: $T \times d_{model}$).

## 6. How It Works
1. A raw text prompt is split into $T$ tokens: $[t_1, t_2, \dots, t_T]$.
2. Each Token ID $t_i$ looks up its row in the Embedding Matrix $\mathbf{E} \in \mathbb{R}^{|V| \times d_{model}}$.
3. The $T$ retrieved embedding vectors are stacked vertically to construct $\mathbf{X}$.

## 7. Required Mathematics
For a sequence of length $T$ and embedding dimension $d_{model}$:

$$\mathbf{X} = \begin{bmatrix} \mathbf{e}_{t_1} \\ \mathbf{e}_{t_2} \\ \vdots \\ \mathbf{e}_{t_T} \end{bmatrix} \in \mathbb{R}^{T \times d_{model}}$$

**Shape Trace:**
- Token IDs: $(T,)$
- Embedding Matrix $\mathbf{E}$: $(|V|, d_{model})$
- Sequence Matrix $\mathbf{X}$: $(T, d_{model})$

## 8. Complete Worked Example
Let sequence length $T = 3$ (Tokens: `"Order"`, `"Shipment"`, `"Receive"`) and $d_{model} = 2$:
- `"Order"` (ID 1) $\to [0.5, -0.2]$
- `"Shipment"` (ID 5) $\to [0.1, 0.9]$
- `"Receive"` (ID 6) $\to [0.8, 0.4]$

$$\mathbf{X} = \begin{bmatrix} 0.5 & -0.2 \\ 0.1 & 0.9 \\ 0.8 & 0.4 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

## 9. Math → Code Mapping
```python
import numpy as np

# Vocabulary size |V|=10, embedding dim d_model=2
E = np.array([
    [0.0, 0.0],  # 0: Pad
    [0.5, -0.2], # 1: Order
    [0.0, 0.0],
    [0.0, 0.0],
    [0.0, 0.0],
    [0.1, 0.9],  # 5: Shipment
    [0.8, 0.4]   # 6: Receive
])

token_ids = [1, 5, 6] # T=3 tokens
X = E[token_ids]      # Array lookup yields Shape (3, 2)
print("Sequence Matrix X shape:", X.shape)
```

## 10. Experiments / What-If Questions
- **What if sequence length $T$ exceeds the model's context window $T_{max}$?**
  The input sequence must be truncated or split into batches of max size $T_{max}$.
- **What if sequences in a batch have different lengths?**
  Shorter sequences are padded with a designated `<PAD>` token ID to match shape $(B, T_{max}, d_{model})$.

## 11. Common Misunderstandings
- **Misunderstanding:** A sequence matrix combines all words by adding or averaging them together into 1 vector.
- **Correction:** No! Averaging creates a single "Bag-of-Words" vector and loses position information. The sequence matrix keeps all $T$ rows distinct (Shape: $T \times d_{model}$).

## 12. Limitations and Trade-Offs
Storing full sequence matrices requires $O(T \cdot d_{model})$ memory per sample. As context window $T$ grows to 128,000 tokens, memory consumption scales linearly before attention layers are applied.

## 13. Where It Appears in the Current Assignment
In **Week 3 Assignment**, your input to the Self-Attention layer will be a sequence matrix $\mathbf{X}$ of shape $(3, 2)$ representing 3 operational log tokens.

## 14. Where It Appears in Modern AI Systems
Every Transformer (GPT-4, Llama-3, Claude) converts prompt tokens into a sequence matrix $\mathbf{X}$ as the very first step before passing data into attention blocks.

## 15. Connection to the Next Concept
Now that we have a sequence matrix $\mathbf{X} \in \mathbb{R}^{T \times d_{model}}$, how do we multiply matrices to compare every token against every other token? That requires **Matrix Transpose** (`20 - MATH - Matrix Transpose and Attention Shapes.md`).

## 16. Teach-Back and Small Application Exercise
If a prompt has $T = 5$ tokens and the model uses $d_{model} = 768$:
1. What is the shape of the sequence matrix $\mathbf{X}$?
2. How many total scalar floating-point numbers are in $\mathbf{X}$?

## 17. Quick Revision Summary
- Single-token models fail because context matters.
- A Sequence Matrix $\mathbf{X} \in \mathbb{R}^{T \times d_{model}}$ stacks token vectors into a 2D matrix.
- Row $i$ corresponds to token position $i$.

## 18. My Understanding
*Fill in your own summary of how sequence matrices represent multi-token context windows.*

## 19. Flashcards
What is the shape of a Sequence Matrix $\mathbf{X}$ for a prompt with $T$ tokens and embedding size $d_{model}$? #card
The shape is $(T, d_{model})$.

Does constructing a Sequence Matrix average the token embeddings together? #card
No. It stacks the $T$ token embeddings vertically as rows, preserving every token's distinct feature vector.

## 20. Sources
- Alammar, J. & Grootendorst, M. *Hands-On Large Language Models*, Chapter 2.
- Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning*, Chapter 12.
