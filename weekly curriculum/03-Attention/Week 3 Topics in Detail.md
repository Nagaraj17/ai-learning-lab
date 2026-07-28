# Week 3 Topics in Detail: Attention & Contextual Representations

The following deep-dive notes in the global `topics/` directory cover Week 3:

1. **[19 - LM - Context Windows and Sequence Representations](../../topics/19%20-%20LM%20-%20Context%20Windows%20and%20Sequence%20Representations.md)**
   - Moving from single tokens to sequence matrix $\mathbf{X} \in \mathbb{R}^{T \times d_{model}}$.

2. **[20 - MATH - Matrix Transpose and Attention Shapes](../../topics/20%20-%20MATH%20-%20Matrix%20Transpose%20and%20Attention%20Shapes.md)**
   - Matrix transpose mechanics $\mathbf{K}^\top$ for inner dimension matching in pairwise dot products ($T \times T$).

3. **[21 - TRANSFORMER - Attention and Contextual Representations](../../topics/21%20-%20TRANSFORMER%20-%20Attention%20and%20Contextual%20Representations.md)**
   - Limitations of static embeddings; dynamic contextual representations $\mathbf{H} = \mathbf{A} \mathbf{V}$.

4. **[22 - TRANSFORMER - Query Key and Value](../../topics/22%20-%20TRANSFORMER%20-%20Query%20Key%20and%20Value.md)**
   - Linear projection weight matrices $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$.

5. **[23 - TRANSFORMER - Scaled Dot-Product Attention](../../topics/23%20-%20TRANSFORMER%20-%20Scaled%20Dot-Product%20Attention.md)**
   - Mathematical formula $\text{Softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}$ and vanishing gradient protection.

6. **[24 - TRANSFORMER - Self-Attention](../../topics/24%20-%20TRANSFORMER%20-%20Self-Attention.md)**
   - Single-head self-attention module implementation.

7. **[25 - TRANSFORMER - Causal Masking](../../topics/25%20-%20TRANSFORMER%20-%20Causal%20Masking.md)**
   - Autoregressive lower-triangular mask setting future score positions to $-\infty$.

8. **[26 - TRANSFORMER - Multi-Head Attention](../../topics/26%20-%20TRANSFORMER%20-%20Multi-Head%20Attention.md)**
   - Parallel attention heads $h$, feature subspace splitting, and output projection $\mathbf{W}_O$.
