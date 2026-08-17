# Week 5 Prerequisite Study-Pack: Tiny Transformer Block from Scratch

Welcome to Week 5! In Week 4, you built Multi-Head Attention—the mechanism that decides **where to look** and **what context to gather**. 

In Week 5, you will assemble the complete machinery that decides **what to do with what it found**: a **Tiny 2-Block Transformer from Scratch in Pure NumPy**.

---

## 📌 Prerequisite Map & Reading Order

Study the concept notes in `topics/` in exact numerical sequence before beginning assignment implementation:

1. [29 - TRANSFORMER - Layer Normalization.md](../../topics/29%20-%20TRANSFORMER%20-%20Layer%20Normalization.md)
2. [30 - TRANSFORMER - Residual Connections and Skip Highways.md](../../topics/30%20-%20TRANSFORMER%20-%20Residual%20Connections%20and%20Skip%20Highways.md)
3. [31 - TRANSFORMER - Feed-Forward Networks and Representation Expansion.md](../../topics/31%20-%20TRANSFORMER%20-%20Feed-Forward%20Networks%20and%20Representation%20Expansion.md)
4. [32 - TRANSFORMER - Stacking Transformer Blocks and Representation Evolution.md](../../topics/32%20-%20TRANSFORMER%20-%20Stacking%20Transformer%20Blocks%20and%20Representation%20Evolution.md)

---

## 🧠 Core Architectural Mental Models

### 1. What Each Component Buys Us

```text
Input Tokens -> Embeddings + PE
                     │
                     ▼
       ┌──────────────────────────┐
       │   Multi-Head Attention   │  <-- Decides WHERE TO LOOK (Context Gathering)
       └─────────────┬────────────┘
                     │
                     ▼
       ┌──────────────────────────┐
       │   Residual Connection    │  <-- Gradient Highway (+1.0 bypass)
       └─────────────┬────────────┘
                     │
                     ▼
       ┌──────────────────────────┐
       │    Layer Normalization   │  <-- Audio Leveler (Mean 0, Variance 1)
       └─────────────┬────────────┘
                     │
                     ▼
       ┌──────────────────────────┐
       │   Feed-Forward Network   │  <-- Decides WHAT TO DO (4x Memory Bank)
       └─────────────┬────────────┘
                     │
                     ▼
       ┌──────────────────────────┐
       │   Residual Connection    │  <-- Gradient Highway (+1.0 bypass)
       └─────────────┬────────────┘
                     │
                     ▼
       ┌──────────────────────────┐
       │    Layer Normalization   │  <-- Audio Leveler (Mean 0, Variance 1)
       └─────────────┬────────────┘
                     │
                     ▼
       [ Transformer Block Output ]  <-- Stack N=2 Blocks for Deep Representation Evolution!
```

---

## 📐 Worked Numerical Examples

### Worked Example 1: Layer Normalization

Given token vector $x = [2.0, 4.0, 6.0, 8.0]$ ($d_{\text{model}}=4$):
1. **Mean ($\mu$)**: $\frac{2+4+6+8}{4} = 5.0$
2. **Variance ($\sigma^2$)**: $\frac{(-3)^2 + (-1)^2 + (1)^2 + (3)^2}{4} = \frac{20}{4} = 5.0 \implies \sigma \approx 2.236$
3. **Normalized Vector ($\hat{x}$)**:
   $$\hat{x} = \left[ \frac{2-5}{2.236}, \; \frac{4-5}{2.236}, \; \frac{6-5}{2.236}, \; \frac{8-5}{2.236} \right] = [-1.3416, \; -0.4472, \; +0.4472, \; +1.3416]$$

---

### Worked Example 2: Feed-Forward Network (FFN)

Given $x = [1.0, 2.0]$ ($d_{\text{model}}=2, d_{\text{ff}}=4$):
1. **$W_1$ Projection**: $z = x @ W_1 = [1.0, 1.0, 0.0, 4.0]$
2. **ReLU Activation**: $h = \text{ReLU}(z) = [1.0, 1.0, 0.0, 4.0]$
3. **$W_2$ Projection**: $\text{FFN}(x) = h @ W_2 = [1.0, 5.0]$

The input vector was non-linearly transformed into $[1.0, 5.0]$!

---

## 🏆 3-Model Generalization Benchmark Setup

In Week 5, we evaluate 3 architectures on an expanded Healthcare GPO Supply Chain dataset (1,000 synthetic log sequences across 15 procurement tokens):

1. **Model A**: Embedding $\rightarrow$ Linear Predictor
2. **Model B**: Embedding $\rightarrow$ Multi-Head Attention $\rightarrow$ Linear Predictor
3. **Model C**: Embedding $\rightarrow$ 2-Block Transformer $\rightarrow$ Linear Predictor

### What to Watch For:
- **Model A** overfits to simple co-occurrence.
- **Model B** handles 1-step attention lookups.
- **Model C** achieves superior generalization on complex multi-step workflows due to block depth ($N=2$) and non-linear FFN memory.
