# Hidden Layers and tanh

## One-line definition
A Hidden Layer defines the internal architecture of the network, and `tanh` is the activation function that bends the linear output into a curve between -1 and 1.

## Why it exists
Without a hidden layer and an activation function, a neural network is just a single straight line (Linear Regression). `tanh` provides the non-linearity needed to solve complex, curved problems.

## Beginner intuition
`Wx + b` draws a straight line. `tanh` takes that straight line and bends it into an 'S' shape. The Hidden layer allows you to combine hundreds of these 'S' shapes together to draw whatever shape you want!

## Week 1 assignment connection
Our Hidden layer has 8 neurons (`hidden_size = 8`). After we calculate `np.dot(x, W1) + b1`, we immediately wrap it in `np.tanh()`.

## Small numerical example
Linear Output: `z = 0.0`  -> `tanh(0.0) = 0.0`
Linear Output: `z = 3.5`  -> `tanh(3.5) ≈ 0.998`
Linear Output: `z = -5.0` -> `tanh(-5.0) ≈ -0.999`

## Common misunderstanding
**Misunderstanding:** `tanh` outputs a probability since it is bounded.
**Correction:** `tanh` outputs a zero-centered value strictly between `-1` and `1`. Probabilities must be strictly positive (0 to 1). `tanh` is an activation, not a probability.

## What happens if removed or changed?
If we remove `tanh`, the equation becomes `(x @ W1 + b1) @ W2 + b2`. Mathematically, a linear equation inside a linear equation collapses into just one single linear equation. The hidden layer becomes entirely useless!

## Teach-back question
If my linear output is `3.0`, the `tanh` output is `0.995`. If the linear output increases heavily to `100.0`, what happens to the `tanh` output?

## Flashcards

What happens mathematically if you remove the activation function from a Neural Network? #card
The entire network collapses into a single linear equation. No matter how many hidden layers you have, a straight line times a straight line is still just a straight line.

What is the output range of the `tanh` activation function? #card
It bounds the output strictly between `-1` and `1`, and is zero-centered. Inputs around +3 or -3 already push the output very close to the extreme bounds.
