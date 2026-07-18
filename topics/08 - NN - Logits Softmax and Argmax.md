# Logits, Softmax, and Argmax

## One-line definition
**Logits** are the raw unrestricted scores, **Softmax** converts them into a probability distribution (0 to 1), and **Argmax** finds the index of the highest probability to make the Prediction.

## Why it exists
To translate the wild, unconstrained numbers produced by a neural network into a clean, human-readable format that tells us how confident the model is about each specific vocabulary word.

## Beginner intuition
- **Logits:** A judge shouting random points: "I give Apple 50 points! Banana -12 points!"
- **Softmax:** A mediator saying: "Okay, Apple has a 98% chance of winning, Banana has a 2% chance."
- **Argmax:** The announcer saying: "The winner is Apple!"

## Week 1 assignment connection
After `W2`, we get a vector of 10 `logits`. We pass them through `softmax(logits)` to get 10 probabilities. Finally, we use `np.argmax(probs)` to find which of the 10 words is the model's actual prediction.

## Small numerical example
Logits: `[3.0, 1.0, 0.1]`
Softmax Probabilities: `[0.84, 0.11, 0.05]` (They sum exactly to 1.0)
Argmax: `0` (Because index 0 has the highest value, 0.84)

## Common misunderstanding
**Misunderstanding:** 100% Model Confidence (Softmax = 1.0) guarantees the answer is correct.
**Correction:** Confidence does not equal correctness. An untrained model can be 100% confident in the completely wrong answer if its random weights happen to align that way.

## What happens if removed or changed?
If we don't use Softmax, we cannot calculate Cross-Entropy Loss, because Cross-Entropy mathematically requires a strict probability (between 0 and 1) to determine how "wrong" the guess was.

## Teach-back question
Why do we subtract the maximum logit value from all other logits before applying the `np.exp()` math inside the Softmax function? (Hint: Numerical stability).

## Flashcards

Are Logits restricted to be between 0 and 1? #card
No. Logits are the raw, unconstrained output scores from the final layer. They can be any number from negative infinity to positive infinity.

What is the difference between Predicted-class confidence and True-class probability? #card
Predicted-class confidence is simply the highest probability from the Softmax output. True-class probability is the probability the network specifically assigned to the *correct* label (which is what we use to calculate Loss).
