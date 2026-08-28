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
| **Model D-no-res** | 2 Blocks without residual connections | Impact of skip-highways on gradient flow |

All models are trained under identical fixed contracts across **5 random seeds** (`[7, 19, 42, 73, 101]`) with early stopping on validation loss.

---

## 📈 Key Evaluation & Training Metrics

### 1. Macro F1-Score
In predicting sequences with rare workflow events (like `APPEAL_OVERTURNED`), standard accuracy can be misleading if the model just predicts the most common event (e.g., `PA_APPROVED`). 
- **Macro F1** computes the F1-score independently for each token class and averages them, treating all classes equally regardless of their frequency in the dataset. This ensures models are penalized if they ignore rare transition events.

### 2. Gradient Clipping
Deep Transformers can experience momentary gradient spikes (e.g., when an attention score aligns perfectly).
- **Gradient Clipping** caps the maximum magnitude of the gradient vector (e.g., $|g| \le 1.0$) during backpropagation. This prevents a single massive parameter update from blowing up the weights and causing $\text{NaN}$ loss.

### 3. FFN Overfitting
While expanding the FFN width ($d_{ff} = 4 d_{model}$) provides more "memory slots" for feature patterns, increasing this width too much (e.g., $8 d_{model}$ or $16 d_{model}$) on a small synthetic dataset will lead to **overfitting**. The model memorizes training samples instead of generalizing, which is why we run a specific ablation to map $d_{ff}$ size against the generalization gap (difference between train and test loss).

---

## 🔬 Experimental Design Principles

### Why One Fixed Dataset Split for Architecture Comparison?
If every architecture trained on a *different* random split, differences in test accuracy could be caused by which *cases* ended up in the test set, not by the architecture itself. By using one fixed split (seed=42) for all models, we isolate the architectural variable.

### Why Separate Initialization Seeds?
Random weight initialization can accidentally make one architecture easier to optimize than another on one run. By using 5 independent seeds, we observe the *distribution* of outcomes, not just one lucky (or unlucky) run.

### The Critical Inferability Rule
Every dataset example follows this rule: **the next event must be inferable from the visible history alone**. If approval depends on `PREV_THERAPY_FAILED`, that event token appears *before* the approval event in the sequence. Violating this rule creates a label-leakage problem where the model cannot possibly predict correctly because the deciding evidence is hidden in the future.

### Holdout Families: Val-only vs Test-only
- **Val-only holdout families** (`step_therapy_exception`, `docs_missing_resubmit_approval`): used during hyperparameter selection and model evaluation on the validation set.
- **Test-only holdout families** (`appeal_overturned`, `contraindication_exception`): never exposed until final test evaluation. These measure genuine generalization to event patterns the model has not seen during training.
- This design ensures that model selection cannot accidentally optimize on the test set's specific scenario patterns.

### What "Evidence-Based Conclusion" Means
A conclusion in this experiment means:
1. We measured a metric (not estimated it).
2. We compared it across multiple seeds (not just one run).
3. We report *what the data shows*, including cases where the hypothesis is **not** supported.
4. We do not claim that a result from 1,200 fictional cases generalizes to all real PA workflows.

