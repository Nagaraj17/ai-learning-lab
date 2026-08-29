# Week 3 Topics in Detail: Attention & Contextual Representations

The following deep-dive notes in the global `topics/` directory cover Week 3:

1. **[19 - LM - Context Windows and Sequence Representations](../../topics/19%20-%20LM%20-%20Context%20Windows%20and%20Sequence%20Representations.md)**
   - Distinguishing the learned Embedding Matrix $\mathbf{E}$ from the computed sequence matrix $\mathbf{X} \in \mathbb{R}^{T \times d_{model}}$.

2. **[20 - TRANSFORMER - Attention and Contextual Representations](../../topics/20%20-%20TRANSFORMER%20-%20Attention%20and%20Contextual%20Representations.md)**
   - The limitations of static embeddings and introducing Attention conceptually.

3. **[21 - TRANSFORMER - Query Key and Value](../../topics/21%20-%20TRANSFORMER%20-%20Query%20Key%20and%20Value.md)**
   - Why we need Query, Key, and Value projections, and pairwise token dot products.

4. **[22 - MATH - Matrix Transpose and Attention Shapes](../../topics/22%20-%20MATH%20-%20Matrix%20Transpose%20and%20Attention%20Shapes.md)**
   - Matrix transpose mechanics $\mathbf{K}^\top$, converting raw scores to weights via Softmax, and computing the Contextual Output $\mathbf{H} = \mathbf{A} \mathbf{V}$.

5. **[23 - TRANSFORMER - Scaled Dot-Product Attention](../../topics/23%20-%20TRANSFORMER%20-%20Scaled%20Dot-Product%20Attention.md)**
   - Score scaling by $\sqrt{d_k}$ to prevent Softmax saturation.

6. **[24 - TRANSFORMER - Self-Attention](../../topics/24%20-%20TRANSFORMER%20-%20Self-Attention.md)**
   - The complete Single-head self-attention flow and its connection to backpropagation training.

7. **[25 - TRANSFORMER - Causal Masking](../../topics/25%20-%20TRANSFORMER%20-%20Causal%20Masking.md)**
   - Autoregressive lower-triangular mask setting future score positions to $-\infty$.

## Boundary Note

Week 3 stops at single-head self-attention plus causal masking.
Multi-Head Attention is intentionally deferred to Week 4 so that the learner
fully understands one head before reasoning about multiple heads, concatenation,
and output projection.
