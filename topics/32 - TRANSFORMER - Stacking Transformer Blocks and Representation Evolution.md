# 32 - TRANSFORMER - Stacking Transformer Blocks and Representation Evolution

## 1. The Problem
A single Transformer block (MHA + FFN + Residuals + LayerNorm) allows tokens to attend to their immediate context and transform their feature representations once.

However, complex language reasoning requires **hierarchical feature abstraction**:
- **Layer 1 (Local Syntax & Adjacency)**: Understands that `Forecast` is preceded by `Inventory`.
- **Layer 2 (Global Context & Disambiguation)**: Synthesizes that the full path `[Receive -> Restock -> Inventory -> Forecast]` implies an upcoming `Order`, whereas the shorter path `[Inventory -> Forecast]` implies an upcoming `Scenario`.

A 1-layer Transformer lacks the depth needed for representations to evolve hierarchically.

---

## 2. Why We Need Something New
We need an architecture that stacks multiple identical Transformer blocks sequentially ($N = 2, 6, 12, 96$).

Each block takes the output vector of the previous block as its input ($X_{l} = \text{Block}_l(X_{l-1})$), allowing token vectors to refine their meanings progressively through layer depth.

---

## 3. One-Line Definition
**Stacking Transformer Blocks** passes feature representations through sequential sub-layer operations, enabling hierarchical representation evolution from low-level local patterns to high-level global context.

---

## 4. Beginner Intuition / Mental Model
Imagine an **Executive Decision Pipeline**:
- **Block 1 (Junior Analyst)**: Reads raw data and extracts basic local facts.
- **Block 2 (Senior Manager)**: Takes the junior analyst's facts, checks multi-step workflow context, and resolves ambiguities.
- **Block 3 (Executive VP)**: Makes the final high-level strategic next-action decision!

---

## 5. What Came Before $\rightarrow$ What Changes Now

| Aspect | 1-Block Transformer | Multi-Block Stacked Transformer ($N=2, 12, 96$) |
| :--- | :--- | :--- |
| **Depth** | Single layer ($N=1$). | Hierarchical depth ($N=2$ to $N=96$). |
| **Representation Capacity** | Limited to 1-hop context gathering. | **Hierarchical Evolution**: Multi-hop reasoning. |
| **Gradient Stability** | Easy. | Requires **Pre-LN + Residual Highways** to prevent vanishing gradients. |

---

## 6. How It Works
For input token sequence $X_0 \in \mathbb{R}^{B \times T \times d_{\text{model}}}$:

1. **Embedding & Positional Encoding**: $X_0 = \text{Embedding}(\text{IDs}) + \text{PE}$.
2. **Block 1**: $X_1 = \text{Block}_1(X_0)$.
3. **Block 2**: $X_2 = \text{Block}_2(X_1)$.
4. **Vocabulary Projection**: $\text{Logits} = X_N W_{\text{vocab}} + b_{\text{vocab}}$.

---

## 7. Required Mathematics

### Formulas:
For block $l \in [1, N]$:
$$X'_{l-1} = X_{l-1} + \text{MHA}(\text{LN}_1(X_{l-1}))$$
$$X_l = X'_{l-1} + \text{FFN}(\text{LN}_2(X'_{l-1}))$$

### Symbol-by-Symbol Breakdown:
- $X_0$: Initial token embedding tensor with shape $(B, T, d_{\text{model}})$.
- $X_{l-1}$: Input tensor to block $l$ with shape $(B, T, d_{\text{model}})$.
- $\text{LN}_1, \text{LN}_2$: Layer Normalization operations.
- $\text{MHA}$: Multi-Head Causal Attention operation.
- $\text{FFN}$: Position-Wise Feed-Forward Network operation.
- $X_l$: Final output tensor of block $l$ with shape $(B, T, d_{\text{model}})$.

---

## 8. Complete Worked Example

Let sequence be $X_0$ with shape $(B=1, T=3, d_{\text{model}}=4)$:
1. **Pass $X_0$ through Block 1**:
   - MHA gathers local 1-step token context $\rightarrow X'_0$.
   - FFN applies non-linear memory retrieval $\rightarrow X_1$.
2. **Pass $X_1$ through Block 2**:
   - MHA attends over Block 1's refined representations, capturing 2-step multi-token relationships $\rightarrow X'_1$.
   - FFN synthesizes final state representations $\rightarrow X_2$.
3. **Project to Logits**:
   - $\text{Logits} = X_2 W_{\text{head}} + b_{\text{head}}$ shape $(1, 3, |V|)$.

---

## 9. Math $\rightarrow$ Code Mapping

```python
# Pass through Stacked Blocks
x = emb + pos_enc
for block in self.blocks:
    x = block.forward(x)

# Project to next-token probabilities
logits = np.matmul(x, self.W_head) + self.b_head
```

---

## 10. Experiments / What-If Questions
- **Does more depth always improve accuracy?** In our Week 5 benchmark on 1,000 cases, Model D-1 ($N=1$) achieved $60.15\%$ accuracy vs Model D ($N=2$) at $57.70\%$. This proves that on small datasets, excessive depth can cause slight over-parameterization.

---

## 11. Common Misunderstandings
- ❌ *Misconception*: "Higher blocks process different tokens than lower blocks."
  - ✅ **Correction**: No! All blocks process the exact same sequence of tokens; what evolves across blocks is the **richness of the feature representation vector** for each token.

---

## 12. Limitations and Trade-Offs
- **Inference Latency**: Sequential block execution increases compute time linearly with depth ($O(N)$).

---

## 13. Where It Appears in the Current Assignment
Evaluated as Model D ($N=2$) vs Model D-1 ($N=1$) in the **Week 5 Generalization Study**.

---

## 14. Where It Appears in Modern AI Systems
- **GPT-2 Small**: 12 blocks ($d_{\text{model}}=768$).
- **GPT-3 175B**: 96 blocks ($d_{\text{model}}=12288$).
- **LLaMA-3 70B**: 80 blocks ($d_{\text{model}}=8192$).

---

## 15. Connection to the Next Concept
Now that you have built a complete 2-block Transformer language model, you are ready for **Week 6 (Tiny GPT & Autoregressive Text Generation)**!

---

## 16. Teach-Back and Small Application Exercise
**Exercise**: Describe how feature representations change from Layer 1 to Layer 12 in a Transformer LLM.

---

## 17. Quick Revision Summary
- Stacking Transformer blocks enables hierarchical feature representation evolution.
- Lower layers capture local syntax; deeper layers capture global semantics.
- Pre-LN + Skip Highways allow scaling to 96+ layers safely.

---

## 18. My Understanding

```markdown
Stacking Transformer blocks allows token vectors to evolve hierarchically. Layer 1 captures simple local relationships, while deeper layers synthesize global multi-step context to make accurate predictions.
```

---

## 19. Flashcards

**Front**: What evolves as signals pass through stacked Transformer blocks?  
**Back**: The feature representation vectors of the tokens, evolving from low-level local syntax to high-level global context.

**Front**: How does compute scaling relate to the number of Transformer blocks $N$?  
**Back**: Compute scale increases linearly $O(N)$ with the number of blocks.

---

## 20. Sources
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
- Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2). OpenAI.
