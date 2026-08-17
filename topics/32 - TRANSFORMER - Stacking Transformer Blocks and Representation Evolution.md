# 32 - TRANSFORMER - Stacking Transformer Blocks and Representation Evolution

## 1. The Problem
A single Transformer block (MHA + FFN + Residuals + LayerNorm) allows tokens to attend to their immediate context and transform their feature representations once.

However, complex language reasoning requires **hierarchical feature abstraction**:
- **Layer 1 (Local Syntax & Adjacency)**: Understands that `Forecast` is preceded by `Inventory`.
- **Layer 2 (Global Context & Disambiguation)**: Synthesizes that the full path `[Receive -> Restock -> Inventory -> Forecast]` implies an upcoming `Order`, whereas the shorter path `[Inventory -> Forecast]` implies an upcoming `Scenario`.

A 1-layer Transformer lacks the depth needed for representations to evolve hierarchically.

---

## 2. Why We Need Something New: Stacking Transformer Blocks
We need an architecture that stacks multiple identical Transformer blocks sequentially ($N = 2, 6, 12, 96$).

Each block takes the output vector of the previous block as its input ($X_{l} = \text{Block}_l(X_{l-1})$), allowing token vectors to refine their meanings progressively through layer depth.

---

## 3. One-Line Definition
**Stacking Transformer Blocks** passes feature representations through sequential sub-layer operations, enabling hierarchical representation evolution from low-level local patterns to high-level global context.

---

## 4. Complete Architecture Blueprint of a 2-Block Transformer

```text
               Input Token IDs (Batch x Seq_Len)
                              │
               [ Token Embedding + Sinusoidal PE ]
                              │
                    Tensor X_0 (B x T x d_model)
                              │
   ======================= BLOCK 1 =======================
   │                                                     │
   │  X_0 ───┬───────────────────────────────┐           │
   │         │                               │           │
   │         ▼                               │ (Residual)│
   │   [ Multi-Head Attention ]              │           │
   │         │                               │           │
   │         ▼                               │           │
   │       MHA(X_0)                          │           │
   │         │                               │           │
   │         └───────────────► (+) ◄─────────┘           │
   │                            │                        │
   │                    [ LayerNorm 1 ]                  │
   │                            │                        │
   │                            ▼                        │
   │                  SubLayer_1 (B x T x d_model)       │
   │                            │                        │
   │  SubLayer_1 ──┬─────────────────────────────┐       │
   │               │                             │       │
   │               ▼                             │(Res)  │
   │         [ Position-Wise FFN (4xd_model) ]   │       │
   │               │                             │       │
   │               ▼                             │       │
   │             FFN(SubLayer_1)                 │       │
   │               │                             │       │
   │               └─────────► (+) ◄─────────────┘       │
   │                            │                        │
   │                    [ LayerNorm 2 ]                  │
   │                            │                        │
   =============================│=========================
                                ▼
                    Tensor X_1 (B x T x d_model)
                                │
   ======================= BLOCK 2 =======================
   │      (Identical Sub-layer Architecture as Block 1)   │
   =============================│=========================
                                ▼
                    Tensor X_2 (B x T x d_model)
                                │
                [ Vocabulary Projection W_vocab ]
                                │
                Next-Token Logits (B x T x Vocab_Size)
```

---

## 5. Python / NumPy Implementation of Stacked Blocks

```python
import numpy as np

def transformer_block(x, mha_weights, ffn_weights, norm_weights):
    """
    Computes a single Transformer Block (Pre-LN variant).
    """
    # Sub-layer 1: Pre-LN Multi-Head Attention + Residual
    norm_1, _ = layer_norm(x, norm_weights['g1'], norm_weights['b1'])
    attn_out, _, _ = forward_multi_head(norm_1, **mha_weights)
    x1 = x + attn_out
    
    # Sub-layer 2: Pre-LN Feed-Forward Network + Residual
    norm_2, _ = layer_norm(x1, norm_weights['g2'], norm_weights['b2'])
    ffn_out, _ = feed_forward_network(norm_2, **ffn_weights)
    x2 = x1 + ffn_out
    
    return x2

def stacked_transformer_model(inputs, embed_table, PE, blocks_weights, W_vocab):
    """
    Passes inputs through Embeddings -> PE -> N Stacked Blocks -> Vocab Logits.
    """
    # 1. Embeddings + PE
    X = embed_table[inputs] + PE
    
    # 2. Sequential Block Pass
    for block_w in blocks_weights:
        X = transformer_block(X, block_w['mha'], block_w['ffn'], block_w['norm'])
        
    # 3. Final Classification Head
    logits = X @ W_vocab
    return logits
```

---

## 6. What Each Component Buys Us (Summary Matrix)

| Component | Function | What It Buys Us |
| :--- | :--- | :--- |
| **Multi-Head Attention** | Context Gathering | Decides **where to look** across multiple subspaces. |
| **Feed-Forward Network** | Feature Processing | Decides **what to do** with gathered context (memory bank). |
| **Residual Connections** | Skip Highways | Eliminates vanishing gradients and preserves input memory. |
| **Layer Normalization** | Activation Leveler | Stabilizes numerical scale (mean 0, variance 1). |
| **Block Depth ($N=2$)** | Representation Evolution | Transforms low-level syntax into high-level global context. |

---

## 7. My Understanding

```markdown
Stacking Transformer blocks sequentially enables representation evolution. Block 1 processes local token relationships, while Block 2 synthesizes deep contextual paths to disambiguate complex workflow predictions.
```

---

## 8. Sources
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
- Radford, A., et al. (2019). *Language Models Are Unsupervised Multitask Learners* (GPT-2).
