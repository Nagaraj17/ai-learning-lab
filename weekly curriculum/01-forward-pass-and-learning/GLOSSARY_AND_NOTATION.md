# Week 1 Glossary and Notation Guide

This guide defines the mathematical symbols and terminology used throughout Week 1.

| Symbol / Term | How to read it aloud | Meaning | Week 1 Shape | Code Location |
|---|---|---|---|---|
| $x$ | "ex" | The input to the network (e.g. a one-hot vector for a single word) | `(1, 10)` | `01_Next_Word_Predictor.ipynb` (Input Data) |
| $y$ | "why" | The true label or target (e.g. the correct next word token ID) | Scalar (e.g., `1`) | `01_Next_Word_Predictor.ipynb` (Target Data) |
| Target | "target" | The "correct answer" the network is trying to predict | N/A | Notebook & Loss functions |
| Label | "label" | Same as Target | N/A | Notebook & Loss functions |
| $W1$ | "double-you one" | The weights connecting the Input Layer to the Hidden Layer | `(10, 8)` | `W1 = np.random.randn(...)` |
| $b1$ | "bee one" | The biases added to the Hidden Layer | `(1, 8)` | `b1 = np.zeros(...)` |
| $z_1$ | "zee one" | The linear transformation before activation ($x \cdot W1 + b1$) | `(1, 8)` | Implicit in `hidden_state = np.tanh(x @ W1 + b1)` |
| $h$ | "aitch" | The activated hidden state (after tanh) | `(1, 8)` | `hidden_state = np.tanh(...)` |
| $W2$ | "double-you two" | The weights connecting the Hidden Layer to the Output Layer | `(8, 10)` | `W2 = np.random.randn(...)` |
| $b2$ | "bee two" | The biases added to the Output Layer | `(1, 10)` | `b2 = np.zeros(...)` |
| Logits | "lo-jits" | The raw, un-normalized output scores of the network | `(1, 10)` | `logits = hidden_state @ W2 + b2` |
| Probabilities | "probabilities" | The normalized output (via Softmax), summing to 1 | `(1, 10)` | `probs = np.exp(...) / np.sum(...)` |
| Loss | "loss" | A single number measuring how wrong the prediction is | Scalar | `loss = -np.log(...)` |
| $d\_$ | "dee" | Prefix meaning "derivative of loss with respect to..." (e.g. $dW1$) | Matches the variable | `dW1`, `db1`, `dW2`, `db2` |
| $\partial$ | "partial" | The partial derivative (e.g. $\frac{\partial loss}{\partial W}$) | N/A | Markdown notes |
| Gradient | "gradient" | The slope telling us how a parameter affects the loss | Matches the parameter | Backward Pass cells |
| Learning Rate | "learning rate" | The multiplier controlling the size of the weight update step | Scalar (e.g., `0.1`) | `learning_rate = 0.1` |
| Epoch | "epoch" | One complete pass of the training data | N/A | `for epoch in range(500):` |
| Shape notation | "shape" | The dimensions of a matrix `(rows, columns)` | N/A | Everywhere |
| Transpose ($^T$) | "transpose" | Flipping a matrix over its diagonal (rows become columns) | N/A | `.T` or `np.transpose()` |
| $\times$ or $*$ | "times" | Element-wise multiplication (multiplying matching positions) | N/A | `*` operator in Python |
| $\cdot$ or $@$ | "dot" | Matrix multiplication / Dot product | N/A | `@` operator or `np.dot()` |
| Outer product | "outer product" | Multiplying a column vector by a row vector to make a matrix | N/A | Used when calculating $dW1$, $dW2$ |
