# Week 5 Prerequisite Study-Pack: Tiny Transformer Block Generalization Study

Welcome to Week 5! In Week 4, you built Multi-Head Attention—the mechanism that decides **where to look** and **what context to gather**. 

In Week 5, you assemble the complete machinery that decides **what to do with what it found**: a **Tiny 2-Block Pre-LN Transformer from Scratch in Pure NumPy** evaluated on a synthetic **Healthcare Step-Therapy Prior-Authorization Workflow Benchmark**.

---

## 📌 Prerequisite Map & Reading Order

Study the concept notes in `topics/` in exact numerical sequence:

1. [29 - TRANSFORMER - Layer Normalization.md](../../topics/29%20-%20TRANSFORMER%20-%20Layer%20Normalization.md)
2. [30 - TRANSFORMER - Residual Connections and Skip Highways.md](../../topics/30%20-%20TRANSFORMER%20-%20Residual%20Connections%20and%20Skip%20Highways.md)
3. [31 - TRANSFORMER - Feed-Forward Networks and Representation Expansion.md](../../topics/31%20-%20TRANSFORMER%20-%20Feed-Forward%20Networks%20and%20Representation%20Expansion.md)
4. [32 - TRANSFORMER - Stacking Transformer Blocks and Representation Evolution.md](../../topics/32%20-%20TRANSFORMER%20-%20Stacking%20Transformer%20Blocks%20and%20Representation%20Evolution.md)

---

## 🧠 Core Architectural Mental Models

### 1. Pre-LN Transformer Block Architecture

```text
               Input Tokens (Batch B x Seq_Len T)
                              │
               [ Token Embedding + Sinusoidal PE ]
                              │
                    Tensor X_0 (B x T x d_model)
                              │
   ======================= BLOCK 1 =======================
   │  X_0 ---+-------------------------------┐           │
   │         │                               │           │
   │         ▼                               │ (Residual)│
   │    [ LayerNorm 1 ]                      │           │
   │         │                               │           │
   │         ▼                               │           │
   │   [ Multi-Head Attention (H=4) ]        │           │
   │         │                               │           │
   │         └───────────────> (+) <─────────┘           │
   │                            │                        │
   │                  SubLayer_1 (B x T x d_model)       │
   │                            │                        │
   │  SubLayer_1 --+-----------------------------┐       │
   │               │                             │       │
   │               ▼                             │ (Res) │
   │          [ LayerNorm 2 ]                    │       │
   │               │                             │       │
   │               ▼                             │       │
   │         [ Position-Wise FFN (4xd_model) ]   │       │
   │               │                             │       │
   │               └─────────> (+) <─────────────┘       │
   =============================│=========================
                                ▼
                    Tensor X_1 (B x T x d_model)
                                │
   ======================= BLOCK 2 =======================
   │     (Identical Sub-layer Architecture as Block 1)   │
   =============================│=========================
                                ▼
                    Tensor X_2 (B x T x d_model)
                                │
                [ Vocabulary Projection W_vocab ]
                                │
                Next-Token Logits (B x T x Vocab_Size)
```

---

## 📐 Worked Numerical Examples

### Worked Example 1: Layer Normalization ($\mu, \sigma^2, \gamma, \beta$)

Given token vector $x = [2.0, 4.0, 6.0, 8.0]$ ($d_{\text{model}}=4$):
1. **Mean ($\mu$)**: $\frac{2+4+6+8}{4} = 5.0$
2. **Variance ($\sigma^2$)**: $\frac{(-3)^2 + (-1)^2 + (1)^2 + (3)^2}{4} = \frac{20}{4} = 5.0 \implies \sigma = \sqrt{5.0 + 1e-5} \approx 2.236$
3. **Normalized Vector ($\hat{x}$)**:
   $$\hat{x} = \left[ \frac{2-5}{2.236}, \; \frac{4-5}{2.236}, \; \frac{6-5}{2.236}, \; \frac{8-5}{2.236} \right] = [-1.3416, \; -0.4472, \; +0.4472, \; +1.3416]$$
4. **Scaled & Shifted Vector ($y = \gamma \hat{x} + \beta$)**:
   If $\gamma = [1, 1, 1, 1]$ and $\beta = [0, 0, 0, 0]$, $y = \hat{x}$.

---

### Worked Example 2: Position-Wise Feed-Forward Network (FFN)

Given $x = [1.0, 2.0]$ ($d_{\text{model}}=2, d_{\text{ff}}=4$):
1. **Projection $\mathbf{W}_1$ ($d_{\text{model}} \to d_{\text{ff}}$)**: $z = x @ \mathbf{W}_1 + \mathbf{b}_1 = [1.0, 1.0, -2.0, 4.0]$
2. **ReLU Activation**: $h = \text{ReLU}(z) = [1.0, 1.0, 0.0, 4.0]$
3. **Projection $\mathbf{W}_2$ ($d_{\text{ff}} \to d_{\text{model}}$)**: $\text{FFN}(x) = h @ \mathbf{W}_2 + \mathbf{b}_2 = [1.0, 5.0]$

The feature representation was expanded non-linearly to process complex state transitions!

---

## 🏆 7-Model Generalization Benchmark Suite

We evaluate 7 model architectures on synthetic healthcare step-therapy prior-authorization workflows (1,000 cases, split 70/15/15 by complete case, with held-out multi-step branch combinations):

| Model ID | Architecture | Claim Tested |
| :--- | :--- | :--- |
| **Model A** | Embedding + Position + Linear Head | Baseline local representation capability |
| **Model B** | Model A + 1 Causal Self-Attention Head | Single contextual view benefit |
| **Model C** | Model A + 4 Causal Self-Attention Heads | Multi-view context gathering benefit |
| **Model D** | Model A + 2 Pre-LN Transformer Blocks | Full Transformer depth, FFN, and normalization |
| **Model D-1** | 1 Pre-LN Transformer Block | Depth contrast ($N=1$ vs $N=2$) |
| **Model D-no-FFN**| 2 Blocks without FFN sub-layers | Impact of non-linear feature processing |
| **Model D-no-LN** | 2 Blocks without LayerNorm | Impact of feature normalization on optimization |

All models are trained under identical fixed contracts across **5 random seeds** (`[7, 19, 42, 73, 101]`) with early stopping on validation loss.
