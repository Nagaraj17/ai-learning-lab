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
2. Under the simplifying assumptions used in the Transformer paper, why do dot-product magnitudes tend to grow with $d_k$, and how can this affect Softmax?

## Part 3: Code & Implementation
1. Write a 3-line NumPy function that creates a $T \times T$ lower-triangular causal mask matrix $\mathbf{M}$ containing $0.0$ for valid past positions and $-1e9$ for future positions.
2. In single-head attention, why do we transpose $\mathbf{K} \in \mathbb{R}^{T \times d_k}$ into $\mathbf{K}^\top \in \mathbb{R}^{d_k \times T}$ before computing $\mathbf{Q}\mathbf{K}^\top$, and what is the output shape?

## Part 4: Debugging & What-If Scenarios
1. **Scenario:** A student inspects an attention matrix $\mathbf{A}$ and notices that `A[2, 0] = 0.50` (token at position 2 attends 50% to earlier token 0), but `A[0, 2] = 0.0` (token at position 0 gives 0% attention to future token 2). Is this a bug in causal masking or expected behavior? Explain why this asymmetry is expected.
2. **Scenario:** After training a tiny single-head attention model for 2 epochs, the attention weights appear completely random. Why should we not expect an untrained or tiny model to produce human-intuitive attention patterns?
