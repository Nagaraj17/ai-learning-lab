# Level 1: Scalar Neuron Training Step

This exercise traces a single complete training step for a highly simplified network that uses only single numbers (scalars) instead of matrices. This isolates the fundamental concepts of Forward Pass, Loss, Backpropagation, and Weight Update without the distraction of multi-dimensional shapes.

## The Goal
Train a single weight ($W$) to correctly predict the target ($y$) given the input ($x$).

## Initialization (Step 0)
- Input: $x = 2.0$
- Target: $y = 10.0$
- Initial Weight (Random): $W = 3.0$
- Learning Rate: $lr = 0.1$

---

## 1. Forward Pass (Prediction)
**Formula:** $prediction = x \times W$
**Meaning:** Calculate the current guess based on the untrained weight.

- Substituted values: $2.0 \times 3.0$
- **Result:** $prediction = 6.0$

---

## 2. Loss Calculation
**Formula:** $loss = \frac{1}{2}(prediction - y)^2$ *(Using standard Mean Squared Error for simplicity here)*
**Meaning:** Measure how wrong the prediction is.

- Substituted values: $\frac{1}{2}(6.0 - 10.0)^2$
- Substituted values: $\frac{1}{2}(-4.0)^2$
- **Result:** $loss = 8.0$

---

## 3. Backpropagation (Gradient Calculation)
**Formula:** $\frac{\partial loss}{\partial W} = (prediction - y) \times x$ *(Chain rule: derivative of loss times derivative of prediction)*
**Meaning:** How much is the weight $W$ to blame for the error? (Slope of the error curve).

- Substituted values: $(6.0 - 10.0) \times 2.0$
- Substituted values: $(-4.0) \times 2.0$
- **Result:** $gradient = -8.0$
- **Interpretation:** A negative gradient means that *increasing* the weight will *decrease* the loss. This tells us we need to step the weight upward.

---

## 4. Parameter Update
**Formula:** $W_{new} = W - (lr \times gradient)$
**Meaning:** Apply the gradient to the weight to improve the network. We subtract the gradient to ensure we step downhill against the slope.

- Substituted values: $3.0 - (0.1 \times -8.0)$
- Substituted values: $3.0 - (-0.8)$
- Substituted values: $3.0 + 0.8$
- **Result:** $W_{new} = 3.8$

---

## 5. Verification (New Prediction and Loss)
Let's run the Forward Pass again with the new weight to prove learning occurred.

- **New Prediction:** $x \times W_{new}$ = $2.0 \times 3.8 = 7.6$ *(Much closer to our target of 10.0!)*
- **New Loss:** $\frac{1}{2}(7.6 - 10.0)^2 = 2.88$ *(The loss dropped significantly from 8.0!)*

We have successfully completed one step of learning.
