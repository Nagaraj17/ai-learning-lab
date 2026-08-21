# 29 - TRANSFORMER - Layer Normalization

## 1. The Problem
In deep neural networks with multiple stacked layers (like 6, 12, 24, or 96 Transformer blocks), feature activations pass through repeated matrix multiplications ($X W_Q, X W_K, X W_V, X W_O, X W_1$). 

As signals move deeper into the network, the numeric magnitude of feature vectors can fluctuate wildly:
- **Exploding Activations**: Vector values grow exponentially ($+0.5 \rightarrow +12.0 \rightarrow +350.0 \rightarrow \text{NaN}$).
- **Vanishing Gradients**: Small weight updates become ineffective because activation scales drift out of proportion.
- **Sensitivity to Learning Rate**: Without scale normalization, training requires delicate learning rate tuning; otherwise, gradient updates cause numeric instability.

In image processing (CNNs), **Batch Normalization (BatchNorm)** solved this by normalizing features across the batch dimension ($B$). However, **BatchNorm fails in Natural Language Processing (NLP)** because sequence lengths vary dynamically, and normalizing across a batch ties independent sentence samples together inappropriately.

---

## 2. Why We Need Something New
We need a normalization technique that:
1. Operates on **each individual sequence token independently** (zero dependence on batch size $B$).
2. Works identically regardless of sequence length $T$ or batch size $B$.
3. Keeps feature activations centered around **mean = 0** and **variance = 1** at every layer.

That mechanism is **Layer Normalization (LayerNorm)**, introduced by Ba, Kiros, and Hinton (2016).

---

## 3. One-Line Definition
**Layer Normalization** standardizes the feature activations of each token vector across its embedding dimensions to have zero mean and unit variance, followed by a learned scaling ($\gamma$) and shifting ($\beta$) transformation.

---

## 4. Beginner Intuition / Mental Model
Imagine an **Automatic Studio Audio Leveler**:
- In a recording room, 4 speakers talk into microphones: one whispers, one screams, one speaks normally.
- **Without LayerNorm**: The screaming speaker blows out the speakers, while the whisperer is lost in noise.
- **With LayerNorm**: The leveler measures each speaker's volume, centers it around 0 decibels, and rescales everyone to a crisp, comfortable, standardized volume level so no single voice overpowers the rest!

---

## 5. What Came Before $\rightarrow$ What Changes Now

| Aspect | Batch Normalization (BatchNorm) | Layer Normalization (LayerNorm) | Root Mean Square Norm (RMSNorm) |
| :--- | :--- | :--- | :--- |
| **Axis of Normalization** | Across Batch Dimension ($B$). | Across Feature Dimension ($d_{\text{model}}$). | Across Feature Dimension ($d_{\text{model}}$) without mean. |
| **NLP Suitability** | Fails due to variable sequence length $T$. | **Ideal for NLP**: Token-independent. | **Modern Favorite**: Faster GPU memory throughput. |
| **Batch Dependency** | High (fails when $B=1$). | **Zero** (works identically for $B=1$). | **Zero** (works identically for $B=1$). |
| **Parameters** | Running mean $\mu_B$, var $\sigma^2_B$. | Mean $\mu$, variance $\sigma^2$, $\gamma, \beta$. | RMS scale, gain $\gamma$. |

---

## 6. How It Works
For a single token vector $x \in \mathbb{R}^{d_{\text{model}}}$:

1. **Calculate Token Mean ($\mu$)**:
   $$\mu = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} x_i$$
2. **Calculate Token Variance ($\sigma^2$)**:
   $$\sigma^2 = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} (x_i - \mu)^2$$
3. **Standardize Vector ($\hat{x}$)**:
   $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
   *(where $\epsilon = 10^{-5}$ prevents division by zero)*
4. **Apply Learned Gain ($\gamma$) and Shift ($\beta$)**:
   $$y_i = \gamma_i \hat{x}_i + \beta_i$$

---

## 7. Required Mathematics

### Formulas:
$$\text{LN}(x) = \gamma \odot \left( \frac{x - \mu(x)}{\sqrt{\sigma^2(x) + \epsilon}} \right) + \beta$$

### Symbol-by-Symbol Breakdown:
- $x$: Input feature vector for one token position with shape $(d_{\text{model}},)$.
- $\mu(x)$: Scalar mean of the elements in vector $x$.
- $\sigma^2(x)$: Scalar variance of the elements in vector $x$.
- $\epsilon$: Small constant ($10^{-5}$) added for numerical stability.
- $\hat{x}$: Zero-mean, unit-variance standardized vector.
- $\gamma$: Learnable gain (scale) vector with shape $(d_{\text{model}},)$, initialized to ones ($1.0$).
- $\beta$: Learnable shift (bias) vector with shape $(d_{\text{model}},)$, initialized to zeros ($0.0$).
- $\odot$: Element-wise (Hadamard) multiplication.

### Tensor Shape Trace:
- Input $X$: $(B, T, d_{\text{model}})$
- $\mu(X)$: $(B, T, 1)$
- $\sigma^2(X)$: $(B, T, 1)$
- Standardized $\hat{X}$: $(B, T, d_{\text{model}})$
- Output $Y$: $(B, T, d_{\text{model}})$

---

## 8. Complete Worked Example

Let a single token vector be $x = [2.0, 4.0, 6.0, 8.0]$ ($d_{\text{model}} = 4$, $\epsilon = 10^{-5}$):

1. **Calculate Mean ($\mu$)**:
   $$\mu = \frac{2.0 + 4.0 + 6.0 + 8.0}{4} = \frac{20.0}{4} = 5.0$$

2. **Calculate Variance ($\sigma^2$)**:
   $$\sigma^2 = \frac{(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2}{4} = \frac{9 + 1 + 1 + 9}{4} = \frac{20}{4} = 5.0$$
   $$\text{std} = \sqrt{5.0 + 1e-5} \approx 2.236068$$

3. **Standardize ($\hat{x}$)**:
   $$\hat{x}_1 = \frac{2.0 - 5.0}{2.236068} = \frac{-3.0}{2.236068} = -1.34164$$
   $$\hat{x}_2 = \frac{4.0 - 5.0}{2.236068} = \frac{-1.0}{2.236068} = -0.44721$$
   $$\hat{x}_3 = \frac{6.0 - 5.0}{2.236068} = \frac{+1.0}{2.236068} = +0.44721$$
   $$\hat{x}_4 = \frac{8.0 - 5.0}{2.236068} = \frac{+3.0}{2.236068} = +1.34164$$

4. **Verify Properties of $\hat{x}$**:
   - Mean of $\hat{x}$: $\frac{-1.34164 - 0.44721 + 0.44721 + 1.34164}{4} = 0.0$
   - Variance of $\hat{x}$: $\frac{(-1.34164)^2 + (-0.44721)^2 + (0.44721)^2 + (1.34164)^2}{4} = 1.0$

5. **Apply $\gamma = [1, 1, 1, 1]$ and $\beta = [0, 0, 0, 0]$**:
   $$y = [-1.34164, \; -0.44721, \; +0.44721, \; +1.34164]$$

---

## 9. Math $\rightarrow$ Code Mapping

```python
import numpy as np

class LayerNormNumPy:
    def __init__(self, d_model, eps=1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self.cache = None

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        x_norm = (x - mean) / std
        out = self.gamma * x_norm + self.beta
        self.cache = (x, x_norm, mean, std)
        return out

    def backward(self, dout):
        x, x_norm, mean, std = self.cache
        self.dgamma = np.sum(dout * x_norm, axis=(0, 1))
        self.dbeta = np.sum(dout, axis=(0, 1))
        dx_norm = dout * self.gamma
        N = self.d_model
        dx = (1.0 / (N * std)) * (
            N * dx_norm 
            - np.sum(dx_norm, axis=-1, keepdims=True) 
            - x_norm * np.sum(dx_norm * x_norm, axis=-1, keepdims=True)
        )
        return dx
```

---

## 10. Experiments / What-If Questions
- **What if $\epsilon = 0$?** If a token has zero variance across features (e.g., $x = [3, 3, 3, 3]$), variance is $0.0$, causing division by zero ($\text{NaN}$). $\epsilon = 10^{-5}$ guarantees numeric safety.
- **What if $\gamma = 0$?** The network zeroes out all feature representations, destroying all information flow.
- **Pre-LN vs Post-LN**:
  - **Post-LN (Original 2017 Transformer)**: $\text{LN}(X + \text{MHA}(X))$. Required careful learning rate warmup to avoid early divergence.
  - **Pre-LN (Modern Standard)**: $X + \text{MHA}(\text{LN}(X))$. The residual connection remains an un-normalized, un-blocked highway for gradients, enabling stable training of 100+ block models.

---

## 11. Common Misunderstandings
- ❌ *Misconception*: "LayerNorm normalizes across the sequence length $T$."
  - ✅ **Correction**: No! LayerNorm normalizes across the feature dimension $d_{\text{model}}$ of each token independently. Each token in a sequence gets its own separate mean $\mu$ and variance $\sigma^2$.
- ❌ *Misconception*: "LayerNorm removes the model's ability to represent large feature values."
  - ✅ **Correction**: No! The learnable parameters $\gamma$ (gain) and $\beta$ (bias) allow the model to restore any feature scale or shift if doing so improves task performance.

---

## 12. Limitations and Trade-Offs
- **Memory Bandwidth Bottleneck**: On modern GPUs, computing mean and variance across feature vectors requires reading memory twice per sub-layer, making LayerNorm memory-bandwidth heavy.
- **RMSNorm Alternative**: LLaMA and Mistral replace LayerNorm with **RMSNorm** ($\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \gamma$), which skips mean subtraction and saves ~10-50% memory bandwidth without loss in model quality.

---

## 13. Where It Appears in the Current Assignment
In the **Week 5 Step-Therapy Benchmark**, LayerNorm is used in Pre-LN configuration:
- Model D-no-LN (removing LayerNorm) resulted in higher test loss ($1.4749 \pm 0.2289$) compared to Model D ($1.3040 \pm 0.1630$), proving that LayerNorm is essential for gradient stabilization.

---

## 14. Where It Appears in Modern AI Systems
- **GPT-2 / GPT-3 / GPT-4**: Uses Pre-LN LayerNorm before MHA and FFN layers.
- **LLaMA 1/2/3 & Mistral**: Uses RMSNorm (Root Mean Square LayerNorm).
- **Vision Transformers (ViT)**: Uses LayerNorm for image patch token vectors.

---

## 15. Connection to the Next Concept
LayerNorm stabilizes activation scales so feature vectors can be safely passed through **Residual Connections (Skip Highways)** without numeric explosions!

---

## 16. Teach-Back and Small Application Exercise
**Exercise**: Given token vector $x = [1.0, 3.0, 5.0, 7.0]$ ($d_{\text{model}}=4$):
1. Calculate the mean $\mu$.
2. Calculate the variance $\sigma^2$.
3. Compute the standardized vector $\hat{x}$.

---

## 17. Quick Revision Summary
- LayerNorm normalizes feature activations across $d_{\text{model}}$ to mean 0 and variance 1 for each token independently.
- It eliminates exploding/vanishing activations in deep Transformer blocks.
- Pre-LN formulation keeps residual skip connections clear for direct gradient propagation.

---

## 18. My Understanding

```markdown
Layer Normalization takes each token's feature vector across d_model, calculates its mean and standard deviation, and rescales the numbers to have mean 0 and variance 1. This prevents activations from blowing up (+50 -> +500 -> NaN) in deep Transformer blocks.
```

---

## 19. Flashcards

**Front**: What axis does Layer Normalization normalize across?  
**Back**: The feature dimension ($d_{\text{model}}$) of each individual token vector independently.

**Front**: Why is Pre-LN preferred over Post-LN in modern Transformers?  
**Back**: Pre-LN places LayerNorm before the sub-layer, allowing the residual skip connection to remain an uninterrupted highway for gradients, improving deep network training stability.

---

## 20. Sources
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization*. arXiv:1607.06450.
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
- Zhang, B., & Sennrich, R. (2019). *Root Mean Square Layer Normalization*. NeurIPS.
