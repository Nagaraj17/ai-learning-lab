# Cross-Entropy Loss

## One-line definition
Cross-Entropy Loss is a mathematical function that measures how wrong a prediction is by looking exclusively at the probability the model assigned to the *true correct class*.

## Why it exists
To mathematically quantify "error." Without a loss function, the network doesn't know if its prediction was amazing or terrible, and therefore doesn't know how to adjust its weights.

## Beginner intuition
Imagine you are taking a multiple-choice test. 
If you are 99% confident the answer is C, and the answer is C, you get a penalty of near 0.
If you are 99% confident the answer is A, but the answer is C, you get a massive penalty! Cross-Entropy heavily punishes models for being *confidently wrong*.

## Week 1 assignment connection
We calculate loss as `loss = -np.log(correct_prob + 1e-9)`. We find the True Token ID, look at the Softmax probability for that specific ID, and calculate the negative log.

## Small numerical example
Formula: `Loss = -log(probability assigned to the true class)`
- If `p(true class) = 0.90`, Loss = `-log(0.90)` ≈ `0.105` (Very small penalty)
- If `p(true class) = 0.50`, Loss = `-log(0.50)` ≈ `0.693` (Moderate penalty)
- If `p(true class) = 0.01`, Loss = `-log(0.01)` ≈ `4.605` (Massive penalty!)

## Common misunderstanding
**Misunderstanding:** Cross-entropy calculates the difference between the highest probability and the second-highest probability.
**Correction:** Cross-entropy only cares about the probability assigned to the *true correct class*. It ignores the Argmax prediction entirely during the loss calculation.

## What happens if removed or changed?
If we don't calculate loss, we cannot perform backpropagation. The training loop dies here.

## Teach-back question
Why do we add `+ 1e-9` (a microscopically small number) when calculating `-np.log(correct_prob)` in the code?

## Flashcards

What specific value does the Cross-Entropy Loss function look at to determine the penalty? #card
It looks exclusively at the True-Class Probability (the probability the model assigned to the correct label), ignoring what the model's actual prediction was.

Why does Cross-Entropy heavily penalize a model for being confidently wrong? #card
Because it uses a logarithmic scale. If the true-class probability is near `0.0` (meaning the model was highly confident in the wrong answer), `-log(0)` explodes towards infinity, creating a massive loss gradient.
