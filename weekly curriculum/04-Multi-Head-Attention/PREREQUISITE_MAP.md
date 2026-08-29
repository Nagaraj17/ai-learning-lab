# PREREQUISITE MAP (Week 4: Multi-Head Attention)

Week 4 continues directly from Week 3. The learner already has the single-head
attention pipeline. The new work starts by making the limitation of that single
learned perspective visible, then introduces multiple independently projected
heads, and only later previews how this fits into a full Transformer block.

## Dependency Story

- Reuse from Week 3:
  - sequence matrix `X`
  - `Q`, `K`, `V`
  - scaled dot-product attention
  - causal masking
  - contextual output from one head
- Learn now:
  - why one head can be limiting
  - how all heads process the same full sequence
  - how separate `W_Q`, `W_K`, `W_V` per head create independent projections
  - how head dimensions relate to `d_model` and head count
  - concatenation and why `W_O` is still needed
  - evidence-based interpretation of specialization, redundancy, and ablation
- Defer:
  - positional encoding details
  - residual connections beyond a preview
  - Layer Normalization beyond a preview
  - feed-forward networks
  - full Transformer block assembly
  - MoE, MQA, GQA, FlashAttention, KV caching

```mermaid
flowchart TD
    subgraph W12 ["Weeks 1-2 Foundations"]
        M04["Vectors, Matrices, Shapes"]
        M05["Matrix Multiplication"]
        N08["Softmax"]
        L14["Embeddings"]
        L15["Embedding Matrix vs Sequence Matrix"]
    end

    subgraph W3 ["Week 3 Reused Understanding"]
        T19["Sequence Matrix X"]
        T20["Why Static Embeddings Need Context"]
        T21["Q, K, V Projections"]
        T22["Attention Score Shapes and K^T"]
        T23["Scaled Dot-Product Attention"]
        T24["Single-Head Self-Attention"]
        T25["Causal Masking"]
    end

    subgraph W4 ["Week 4 Required Now"]
        W4A["Visible Limitation of One Head"]
        W4B["Same Sequence Enters Every Head"]
        W4C["Independent Per-Head W_Q, W_K, W_V"]
        W4D["Head Count, d_model, d_k, d_v"]
        W4E["Per-Head Scaled Dot-Product Attention"]
        W4F["Per-Head Outputs and Attention Maps"]
        W4G["Concatenation Across Heads"]
        W4H["Output Projection W_O"]
        W4I["Complete Multi-Head Forward Pass"]
        W4J["Specialization, Redundancy, Ablation"]
    end

    subgraph Later ["Deferred Beyond Week 4"]
        D1["Positional Encoding"]
        D2["Residual Connections (full treatment)"]
        D3["Layer Normalization (full treatment)"]
        D4["Feed-Forward Network"]
        D5["Full Transformer Block"]
        D6["MoE / MQA / GQA / FlashAttention"]
    end

    M04 --> T22
    M05 --> T21
    M05 --> T22
    N08 --> T23
    L14 --> T20
    L15 --> T19

    T19 --> T21
    T20 --> T21
    T21 --> T22
    T22 --> T23
    T23 --> T24
    T24 --> T25

    T24 --> W4A
    T25 --> W4E
    W4A --> W4B
    W4B --> W4C
    W4C --> W4D
    W4D --> W4E
    W4E --> W4F
    W4F --> W4G
    W4G --> W4H
    W4H --> W4I
    W4I --> W4J

    W4I --> D2
    W4I --> D3
    W4I --> D4
    D1 --> D5
    D2 --> D5
    D3 --> D5
    D4 --> D5
    W4J --> D6
```
