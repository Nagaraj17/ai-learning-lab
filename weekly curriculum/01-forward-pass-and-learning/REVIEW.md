# Review: Week 1 (Forward Pass & Learning)

Before proceeding to Week 2, ensure you can confidently answer the following questions without checking the notes. 
*(If you are unsure of any answer, consult the `topics/` directory).*

---

## 1. Conceptual Knowledge
1. **Why do we use One-Hot encoding instead of raw Token IDs?**
2. **What is the mathematical purpose of the `tanh` activation function?**
3. **What is the difference between Logits and Softmax probabilities?**
4. **Why do we subtract the maximum logit before applying Softmax?**
5. **How does Cross-Entropy Loss penalize confidently wrong predictions?**
6. **Why does achieving 100% accuracy on the PO Lifecycle problem not prove that our model can generalize?**

---

## 2. Code Comprehension
Look at the following lines of code from the assignment. What exactly do they do?

1. `logits = hidden_state @ W2 + b2`
2. `probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)`
3. `loss = -np.log(probs[np.arange(len(y)), y] + 1e-9)`
4. `d_logits = probs.copy(); d_logits[np.arange(len(y)), y] -= 1`
5. `W1 -= learning_rate * dW1`

---

## 3. Mathematical Calculations
Without using code, calculate the following:

1. **Matrix Shapes:** If your input $X$ is `(1, 10)` and your weight matrix $W$ is `(10, 8)`, what is the shape of the output?
2. **Gradient Sign:** If the gradient for a weight is $+1.5$, do we increase or decrease the weight to reduce the loss?
3. **One-Hot Selection:** If $X = [0, 0, 1]$ and $W$ is a $3 \times 3$ matrix, which row of $W$ does the multiplication $X \cdot W$ select?

---

## 4. Debugging & What-If Scenarios
What would happen to the neural network if you made these changes?

1. You initialize `W1` and `W2` to all zeros instead of using `np.random.randn()`.
2. You set the `learning_rate` to `0.0`.
3. You set the `learning_rate` to `100.0`.
4. You completely remove the `tanh` activation step.
