# Cross-Entropy Loss

## One-line definition
Cross-Entropy Loss is a mathematical function that measures how wrong a prediction is by looking exclusively at the probability the model assigned to the *true correct class*.

## Why it exists
To mathematically quantify "error." Without a loss function, the network doesn't know if its prediction was amazing or terrible, and therefore doesn't know how to adjust its weights.

## Beginner intuition
Imagine you are taking a multiple-choice test. 
If you are 99% confident the answer is C, and the answer is C, you get a penalty of near 0.
If you are 99% confident the answer is A, but the answer is C, you get a massive penalty! Cross-Entropy heavily punishes models for being *confidently wrong*.

## The Mathematical Formula
For a single example in multi-class classification, the formula is:

$$ Loss = - \sum_{c=1}^{M} y_{o,c} \log(p_{o,c}) $$

Where:
- $M$ is the number of classes (vocabulary size).
- $y$ is the true label (1 for the correct class, 0 for all others).
- $p$ is the predicted probability for that class.

**The Simplification:**
Because $y$ is $0$ for every incorrect class, multiplying anything by $0$ makes it disappear. The entire sum collapses into just one term — the one for the true class:

$$ Loss = - \log(p_{\text{true\_class}}) $$

## The Loss Penalty Curve (Table)
Notice how the penalty explodes exponentially as the model becomes more confident in the wrong answer:

| True-Class Probability ($p$) | Loss ($-log(p)$) | Interpretation |
|----------------|----------------|----------------|
| `0.99`         | `0.01`         | Almost perfect. Very small penalty. |
| `0.90`         | `0.11`         | Very good. Small penalty. |
| `0.50`         | `0.69`         | Unsure. Moderate penalty. |
| `0.10`         | `2.30`         | Very wrong. High penalty. |
| `0.01`         | `4.61`         | Confidently wrong. Massive penalty! |
| `0.0001`       | `9.21`         | Completely wrong. Extreme penalty! |

## Week 1 assignment connection
We calculate loss as `loss = -np.log(correct_prob + 1e-9)`. We find the True Token ID, look at the Softmax probability for that specific ID, and calculate the negative log.

## Small numerical example
- If `p(true class) = 0.90`, Loss = `-log(0.90)` ≈ `0.105`
- If `p(true class) = 0.50`, Loss = `-log(0.50)` ≈ `0.693`
- If `p(true class) = 0.01`, Loss = `-log(0.01)` ≈ `4.605`

## Common misunderstanding
**Misunderstanding:** Cross-entropy calculates the difference between the highest probability and the second-highest probability.
**Correction:** Cross-entropy only cares about the probability assigned to the *true correct class*. It ignores the Argmax prediction entirely during the loss calculation.

## What happens if removed or changed?
If we don't calculate loss, we cannot perform backpropagation. The training loop dies here.

## Teach-back question
Why do we add `+ 1e-9` (a microscopically small number) when calculating `-np.log(correct_prob)` in the code?

## My Understanding
*(Learner space to explain this in their own words later)*.

## Exercises
1. If the vocabulary is `["Cat", "Dog", "Bird"]`, the true label is "Bird", and the model outputs probabilities `[0.70, 0.20, 0.10]`, what number is passed into the `-log()` function to calculate the loss?
2. Why is `-log(0)` mathematically dangerous in a program?

## Flashcards

What specific value does the Cross-Entropy Loss function look at to determine the penalty? #card
It looks exclusively at the True-Class Probability (the probability the model assigned to the correct label), ignoring what the model's actual prediction was.

Why does Cross-Entropy heavily penalize a model for being confidently wrong? #card
Because it uses a logarithmic scale. If the true-class probability is near `0.0` (meaning the model was highly confident in the wrong answer), `-log(0)` explodes towards infinity, creating a massive loss gradient.
