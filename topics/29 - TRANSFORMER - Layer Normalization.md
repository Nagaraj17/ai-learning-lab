# 29 - TRANSFORMER - Layer Normalization

## 1. The Problem
In deep neural networks with multiple stacked layers (like 6, 12, or 96 Transformer blocks), feature activations pass through repeated matrix multiplications ($X W_Q, X W_K, X W_V, X W_O, X W_1$). 

As signals move deeper into the network, the numeric magnitude of vectors can fluctuate wildly:
- **Exploding Activations**: Vector numbers grow exponentially ($+0.5 \rightarrow +12.0 \rightarrow +350.0 \rightarrow \text{NaN}$).
- **Vanishing Gradients**: Small weight updates become ineffective because activation scales drift out of proportion.
- **Sensitivity to Learning Rate**: Without scale normalization, training requires delicate learning rate tuning; otherwise, gradient updates cause numeric instability.

In image processing (CNNs), **Batch Normalization (BatchNorm)** solved this by normalizing features across the batch dimension. However, **BatchNorm fails in Natural Language Processing (NLP)** because sequence lengths vary dynamically, and normalizing across a batch ties independent sentence samples together inappropriately.

---

## 2. Why We Need Something New: Layer Normalization
We need a normalization technique that:
1. Operates on **each individual sequence token independently** (zero dependence on batch size).
2. Works identically regardless of sequence length or batch size.
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

| Component | Batch Normalization (BatchNorm) | Layer Normalization (LayerNorm) |
| :--- | :--- | :--- |
| **Axis of Normalization** | Normalizes across the **Batch Dimension** ($B$). | Normalizes across the **Feature Dimension** ($d_{\text{model}}$). |
| **NLP Suitability** | Fails due to variable sequence lengths and batch dependence. | **Ideal for NLP**: Each token is normalized independently. |
| **Dependency** | Dependent on batch size (fails when batch size = 1). | 100% Independent of batch size (works for $B=1$). |

---

## 6. How It Works: Step-by-Step

For a single token embedding vector $x \in \mathbb{R}^{d_{\text{model}}}$:

1. **Calculate Token Mean ($\mu$)**:
   Compute the average value across all $d_{\text{model}}$ feature dimensions:
   $$\mu = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} x_i$$

2. **Calculate Token Variance ($\sigma^2$)**:
   Compute the average squared deviation from the mean:
   $$\sigma^2 = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} (x_i - \mu)^2$$

3. **Standardize Features ($\hat{x}$)**:
   Subtract mean and divide by standard deviation (with a small constant $\epsilon = 10^{-5}$ to prevent division by zero):
   $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

4. **Learnable Scale & Shift ($\gamma$ and $\beta$)**:
   Multiply by a learnable gain parameter $\gamma \in \mathbb{R}^{d_{\text{model}}}$ and add a learnable bias parameter $\beta \in \mathbb{R}^{d_{\text{model}}}$:
   $$y_i = \gamma_i \cdot \hat{x}_i + \beta_i$$

---

## 7. Pre-LN vs Post-LN Architecture

In Transformer blocks, LayerNorm can be placed in two positions:

1. **Post-LN (Original 2017 Transformer Paper)**:
   $$x_{l+1} = \text{LayerNorm}\Big(x_l + \text{SubLayer}(x_l)\Big)$$
   - *Drawback*: Requires careful learning rate warm-up schedules to prevent early gradient instability.

2. **Pre-LN (Modern Standard: GPT-2, GPT-3, LLaMA)**:
   $$x_{l+1} = x_l + \text{SubLayer}\Big(\text{LayerNorm}(x_l)\Big)$$
   - *Advantage*: Gradients flow directly through the main residual spine without passing through LayerNorm parameter constraints, making deep model training significantly more stable!

---

## 8. Complete Worked Example (Small Numbers)

Consider a single token vector with $d_{\text{model}} = 4$:
$$x = [2.0, \; 4.0, \; 6.0, \; 8.0]$$

Suppose $\gamma = [1.0, 1.0, 1.0, 1.0]$ and $\beta = [0.0, 0.0, 0.0, 0.0]$, $\epsilon = 0.0$.

1. **Mean ($\mu$)**:
   $$\mu = \frac{2 + 4 + 6 + 8}{4} = \frac{20}{4} = 5.0$$

2. **Variance ($\sigma^2$)**:
   $$\sigma^2 = \frac{(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2}{4} = \frac{9 + 1 + 1 + 9}{4} = \frac{20}{4} = 5.0$$
   $$\sigma = \sqrt{5.0} \approx 2.236$$

3. **Normalized Vector ($\hat{x}$)**:
   - $\hat{x}_1 = (2 - 5) / 2.236 = -3 / 2.236 \approx -1.3416$
   - $\hat{x}_2 = (4 - 5) / 2.236 = -1 / 2.236 \approx -0.4472$
   - $\hat{x}_3 = (6 - 5) / 2.236 = +1 / 2.236 \approx +0.4472$
   - $\hat{x}_4 = (8 - 5) / 2.236 = +3 / 2.236 \approx +1.3416$

$$\hat{x} = [-1.3416, \; -0.4472, \; +0.4472, \; +1.3416]$$

*Verification*: Mean of $\hat{x}$ is $0.0$, Variance of $\hat{x}$ is $1.0$!

---

## 9. Python / NumPy Implementation

```python
import numpy as np

def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    """
    Computes Layer Normalization on input tensor x of shape (Batch, Seq_Len, d_model).
    """
    d_model = x.shape[-1]
    if gamma is None:
        gamma = np.ones(d_model)
    if beta is None:
        beta = np.zeros(d_model)
        
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    
    x_hat = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_hat + beta
    
    cache = (x, x_hat, mean, var, gamma, beta, eps)
    return out, cache

# Quick Test
x_test = np.array([[[2.0, 4.0, 6.0, 8.0]]])
out_test, _ = layer_norm(x_test)
print("Normalized Vector:", out_test)
print("Mean:", np.mean(out_test, axis=-1))
print("Variance:", np.var(out_test, axis=-1))
```

---

## 10. Where It Appears in Modern AI Systems
- **GPT-2 / GPT-3 / GPT-4**: Uses Pre-LN before Attention and FFN sub-layers.
- **LLaMA / LLaMA-2 / LLaMA-3**: Uses **RMSNorm** (Root Mean Square Normalization), a simplified variant of LayerNorm that omits mean subtraction to save GPU memory bandwidth.

---

## 11. My Understanding

```markdown
Layer Normalization takes each token's feature vector across d_model, calculates its mean and standard deviation, and rescales the numbers to have mean 0 and variance 1. This prevents activations from blowing up (+50 -> +500 -> NaN) in deep Transformer blocks.
```

---

## 12. Flashcards

**Front**: What is the key difference between Batch Normalization and Layer Normalization?  
**Back**: Batch Normalization normalizes across the batch dimension ($B$) and depends on batch size, whereas Layer Normalization normalizes across the feature dimension ($d_{\text{model}}$) of each token independently.

**Front**: Why is Pre-LN preferred over Post-LN in modern Transformers?  
**Back**: Pre-LN places LayerNorm before the sub-layer, allowing the residual skip connection to remain an uninterrupted highway for gradients, improving deep network training stability.

---

## 13. Sources
- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization*. arXiv:1607.06450.
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS.
