# Level 2: Complete Training Step (Parameter Update)

This exercise applies the gradients calculated in `03_MANUAL_BACKPROPAGATION.md` to update the network's weights, proving that the loss decreases.

## Stored Values
- **Learning Rate ($lr$)**: `0.1`

**Layer 2 Weights ($W2$) and Gradients ($dW2$)**
- $W2$: `[[0.4, 0.1], [-0.3, 0.6]]`
- $dW2$: `[[0.2673, -0.2673], [-0.1142, 0.1142]]`

---

## 1. Parameter Update (Layer 2)
**Formula:** $W2_{new} = W2 - (lr \times dW2)$
**Meaning:** Subtract a fraction of the gradient to step downhill.

- $lr \times dW2 = [[0.0267, -0.0267], [-0.0114, 0.0114]]$
- $W2_{new}$:
  ```
  [[ 0.4000 - 0.0267,  0.1000 - (-0.0267)],
   [-0.3000 - (-0.0114), 0.6000 - 0.0114]]
  ```
- **Result:** 
  ```
  [[ 0.3733,  0.1267],
   [-0.2886,  0.5886]]
  ```

*Let's analyze this:* The network needed to increase the probability for class 1. W2's top-right weight connected a positive hidden activation ($0.4621$) to class 1. That weight *increased* from $0.1$ to $0.1267$, which will successfully push the score for class 1 higher next time!

---

## 2. Parameter Update (Layer 1)
**Formula:** $W1_{new} = W1 - (lr \times dW1)$

- $W1$: `[[0.5, -0.2], [-0.1, 0.3]]`
- $dW1$: `[[0.1365, -0.5002], [0.0000, 0.0000]]`
- $lr \times dW1 = [[0.0137, -0.0500], [0.0000, 0.0000]]$
- **Result:**
  ```
  [[ 0.4863, -0.1500],
   [-0.1000,  0.3000]]
  ```

---

## 3. Verification
If we run a Forward Pass again using $W1_{new}$ and $W2_{new}$:
- **New Probabilities:** `[0.5593, 0.4407]`
- **Old Probabilities:** `[0.5784, 0.4216]`

The probability assigned to the *correct* class (`1`) went up from $42.16\%$ to $44.07\%$.
The loss dropped from $0.8637$ to $0.8193$.

**Learning occurred successfully.**

---

## Learner Workspace

### Predict the Sign
Look at the Layer 1 update for `W1`:
- $W1$: `[[0.5, -0.2], [-0.1, 0.3]]`
- $dW1$: `[[0.1365, -0.5002], [0.0000, 0.0000]]`
- The gradient for the top-right weight ($-0.2$) is **negative** ($-0.5002$).
- Based on the update formula $W - (lr \times grad)$, will the new weight be more positive or more negative than $-0.2$?

### Blank Calculation Table
Try calculating a simple parameter update step:
- $W$: `[[0.8, -0.5], [0.2, 0.9]]`
- $grad$: `[[0.5, 0.1], [-0.4, 0.0]]`
- $lr$: `0.1`

| Weight Index | Current W | Gradient | Calculation $W - (lr \times grad)$ | New W |
|--------------|-----------|----------|------------------------------------|-------|
| Top-Left     | `0.8`     | `0.5`    | | |
| Top-Right    | `-0.5`    | `0.1`    | | |
| Bottom-Left  | `0.2`     | `-0.4`   | | |
| Bottom-Right | `0.9`     | `0.0`    | | |
