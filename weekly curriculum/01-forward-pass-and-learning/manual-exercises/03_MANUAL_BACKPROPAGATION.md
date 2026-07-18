# Level 2: Manual Backpropagation

This exercise traces the gradients backward through the tiny network from `02_MANUAL_FORWARD_PASS.md`.

## Stored Values from Forward Pass
- **True Label:** `1`
- **$x$:** `[1, 0]`
- **$h_1$:** `[0.4621, -0.1974]`
- **$probs$:** `[0.5784, 0.4216]`
- **$W2$:** 
  ```
  [[ 0.4,  0.1],
   [-0.3,  0.6]]
  ```

---

## 1. Gradient at Output (Logits)
**Formula:** $d\_logits = probs - true\_one\_hot$
**Meaning:** The derivative of Cross-Entropy combined with Softmax simplifies beautifully to just (Prediction - Target).

- True one-hot: `[0, 1]` (Because the target label is 1)
- $d\_logits = [0.5784, 0.4216] - [0, 1]$
- **Result:** $d\_logits = [0.5784, -0.5784]$ (Shape: `(2,)`)
- *Interpretation:* The network predicted class 0 too high (positive gradient), and class 1 too low (negative gradient).

## 2. Gradients for Layer 2 Weights & Bias
**Formula ($dW2$):** $dW2 = h_1^T \cdot d\_logits$ (Outer product)
**Meaning:** How much did W2 contribute to the error at the logits? (Hidden activations $\times$ logit errors).

- $h_1$ shape `(2,)` treated as column `(2, 1)`: `[[0.4621], [-0.1974]]`
- $d\_logits$ shape `(2,)` treated as row `(1, 2)`: `[[0.5784, -0.5784]]`
- **Result ($dW2$):**
  ```
  [[ (0.4621 * 0.5784),  (0.4621 * -0.5784)],
   [(-0.1974 * 0.5784), (-0.1974 * -0.5784)]]
  
  [[ 0.2673, -0.2673],
   [-0.1142,  0.1142]]
  ```
  (Shape: `(2, 2)`)

**Formula ($db2$):** $db2 = d\_logits$
- **Result:** $db2 = [0.5784, -0.5784]$ (Shape: `(2,)`)

## 3. Gradient at Hidden Layer
**Formula:** $d\_hidden = d\_logits \cdot W2^T$
**Meaning:** Pass the blame backward through the W2 weights to find how much the hidden layer is at fault.

- $W2^T$ (Transpose of W2):
  ```
  [[ 0.4, -0.3],
   [ 0.1,  0.6]]
  ```
- $d\_hidden = [0.5784, -0.5784] \cdot W2^T$
- Node 0: $(0.5784 \times 0.4) + (-0.5784 \times 0.1) = 0.2314 - 0.0578 = 0.1736$
- Node 1: $(0.5784 \times -0.3) + (-0.5784 \times 0.6) = -0.1735 - 0.3470 = -0.5205$
- **Result:** $d\_hidden = [0.1736, -0.5205]$ (Shape: `(2,)`)

## 4. Gradient through Activation (tanh)
**Formula:** $d\_z1 = d\_hidden \times (1 - h_1^2)$ (Element-wise multiplication)
**Meaning:** Adjust the blame by the derivative of tanh. If the neuron was saturated (near 1 or -1), the derivative is near 0, meaning it can't learn much.

- $h_1^2 = [0.4621^2, -0.1974^2] = [0.2135, 0.0390]$
- $(1 - h_1^2) = [0.7865, 0.9610]$
- $d\_z1 = [0.1736 \times 0.7865, -0.5205 \times 0.9610]$
- **Result:** $d\_z1 = [0.1365, -0.5002]$ (Shape: `(2,)`)

## 5. Gradients for Layer 1 Weights & Bias
**Formula ($dW1$):** $dW1 = x^T \cdot d\_z1$
**Meaning:** How much did W1 contribute to the error at the hidden layer?

- $x = [1, 0]$ (Column: `[[1], [0]]`)
- $d\_z1 = [[0.1365, -0.5002]]$
- **Result ($dW1$):**
  ```
  [[ 0.1365, -0.5002],
   [ 0.0000,  0.0000]]
  ```
  *(Notice the second row is zero because the second input was 0. Those weights contributed nothing to this specific error!)*

**Formula ($db1$):** $db1 = d\_z1$
- **Result:** $db1 = [0.1365, -0.5002]$ (Shape: `(2,)`)
