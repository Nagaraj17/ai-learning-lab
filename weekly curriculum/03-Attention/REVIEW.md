# REVIEW (Week 3: Attention)

Test your conceptual, mathematical, and code mastery of Week 3 before completing the module:

## Part 1: Conceptual Understanding
1. Why do static embeddings fail on words with multiple contextual meanings (e.g. `"bank"`)?
2. In Self-Attention, what are the distinct functional roles of Query ($\mathbf{Q}$), Key ($\mathbf{K}$), and Value ($\mathbf{V}$)?
3. Why does autoregressive language modeling require Causal Masking?
4. Explain why Attention weights must NOT automatically be interpreted as definitive explanations of model decision making (Jain & Wallace, 2019).

## Part 2: Mathematical & Shape Tracing
1. Given sequence length $T = 4$, embedding size $d_{model} = 8$, key size $d_k = 4$, and value size $d_v = 4$:
   - What is the shape of sequence matrix $\mathbf{X}$?
   - What is the shape of projection weights $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$?
   - What is the shape of raw score matrix $\mathbf{S} = \mathbf{Q} \mathbf{K}^\top$?
   - What is the shape of contextual output $\mathbf{H} = \mathbf{A} \mathbf{V}$?
2. Why do we scale raw scores by $\frac{1}{\sqrt{d_k}}$ before passing them to Softmax? What happens to Softmax gradients if we omit this scale factor for $d_k = 128$?

## Part 3: Code & Implementation
1. Write a 3-line NumPy function that creates a $T \times T$ lower-triangular causal mask matrix $\mathbf{M}$ containing $0.0$ for valid past positions and $-1e9$ for future positions.
2. In PyTorch, what is the dimension permutation order required when splitting $Q \in \mathbb{R}^{B \times T \times d_{model}}$ into multi-head shape $(B, h, T, d_k)$?

## Part 4: Debugging & What-If Scenarios
1. **Scenario:** A student notices that token 1 (`"Order"`) receives $50\%$ attention from token 3 (`"Receive"`), but token 3 receives $0\%$ attention from token 1. Is this a bug in causal masking or expected behavior? Explain why.
2. **Scenario:** After training a tiny single-head attention model for 2 epochs, the attention weights appear completely random. Why should we not expect an untrained or tiny model to produce human-intuitive attention patterns?
