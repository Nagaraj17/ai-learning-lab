# Level 2: Manual Forward Pass

This exercise traces the forward pass of a tiny neural network: 2 inputs, 2 hidden neurons, 2 output classes. We round to 4 decimal places.

## Initialization
- **Input ($x$)**: `[1, 0]` (Shape: `(2,)`) - E.g., Token ID 0.
- **Target Label**: `1` (We want to predict class 1).

**Layer 1 (Hidden Layer)**
- $W1$ (Shape `(2, 2)`): 
  ```
  [[ 0.5, -0.2],
   [-0.1,  0.3]]
  ```
- $b1$ (Shape `(2,)`): `[0.0, 0.0]`

**Layer 2 (Output Layer)**
- $W2$ (Shape `(2, 2)`): 
  ```
  [[ 0.4,  0.1],
   [-0.3,  0.6]]
  ```
- $b2$ (Shape `(2,)`): `[0.0, 0.0]`

---

## 1. Linear Transformation 1
**Formula:** $z_1 = x \cdot W1 + b1$
**Meaning:** Route inputs through the first set of weights.

- $x \cdot W1 = [1, 0] \cdot W1 = [0.5, -0.2]$ *(Because $x$ is one-hot, it just selects the first row of W1).*
- $z_1 = [0.5, -0.2] + [0.0, 0.0] = [0.5, -0.2]$
- **Result:** $z_1 = [0.5000, -0.2000]$ (Shape: `(2,)`)

## 2. Activation 1 (tanh)
**Formula:** $h_1 = \tanh(z_1)$
**Meaning:** Add non-linearity.

- $h_1 = [\tanh(0.5), \tanh(-0.2)]$
- **Result:** $h_1 = [0.4621, -0.1974]$ (Shape: `(2,)`)

## 3. Linear Transformation 2 (Logits)
**Formula:** $logits = h_1 \cdot W2 + b2$
**Meaning:** Route hidden activations to the final output scores.

- $h_1 \cdot W2 = [0.4621, -0.1974] \cdot W2$
- Logit 0: $(0.4621 \times 0.4) + (-0.1974 \times -0.3) = 0.1848 + 0.0592 = 0.2440$
- Logit 1: $(0.4621 \times 0.1) + (-0.1974 \times 0.6) = 0.0462 - 0.1184 = -0.0722$
- **Result:** $logits = [0.2440, -0.0722]$ (Shape: `(2,)`)

## 4. Output Activation (Softmax)
**Formula:** $probs = \frac{e^{logits}}{\sum e^{logits}}$
**Meaning:** Convert raw scores into a probability distribution.

- $e^{logits} = [e^{0.2440}, e^{-0.0722}] = [1.2763, 0.9303]$
- Sum of $e^{logits} = 1.2763 + 0.9303 = 2.2066$
- $probs = [\frac{1.2763}{2.2066}, \frac{0.9303}{2.2066}]$
- **Result:** $probs = [0.5784, 0.4216]$ (Shape: `(2,)`)

## 5. Cross-Entropy Loss
**Formula:** $loss = -\log(probs[true\_label])$
**Meaning:** Measure error based on the true class probability.

- True label is `1`.
- $probs[1] = 0.4216$
- **Result:** $loss = -\log(0.4216) = 0.8637$ (Shape: Scalar)
