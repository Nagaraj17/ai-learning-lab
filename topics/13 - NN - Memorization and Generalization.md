# Memorization and Generalization

## One-line definition
**Memorization** is when the network perfectly learns the exact training examples, while **Generalization** is when the network understands the underlying logic well enough to solve examples it has never seen before.

## Why it exists
To evaluate if a model is actually useful. A self-driving car that memorizes one specific track perfectly (but crashes on a new road) is useless. It must generalize.

## Beginner intuition
Memorization is cramming for a math test by memorizing the answers to the practice sheet. You'll get 100% on the practice sheet, but 0% on the real test. Generalization is actually learning the formulas.

## Week 1 assignment connection
Our Week 1 task (predicting the PO Lifecycle loop) is purely a memorization task! Because the loop is a fixed sequence of 10 words repeating infinitely, there are no "new" or "unseen" examples to test on. The model is just acting as a lookup table.

## Small numerical example
- Memorization: 100% accuracy on Training Data.
- Generalization: 95% accuracy on Unseen Testing Data.

## Common misunderstanding
**Misunderstanding:** 100% training accuracy means the model is perfect.
**Correction:** 100% training accuracy often means the model has "overfit" the data (memorized it). We don't actually care about training accuracy; we care about testing accuracy.

## What happens if removed or changed?
If we do not test for generalization (using a separate test dataset), we might deploy a completely broken model into the real world, thinking it is perfect.

## Teach-back question
Why is the Week 1 Purchase Order Lifecycle problem incapable of proving that our neural network can generalize?

## Flashcards

What is the difference between Memorization and Generalization in Machine Learning? #card
Memorization is achieving high accuracy on the exact data used for training. Generalization is the ability to achieve high accuracy on completely new, unseen data.

Why does our Week 1 PO Lifecycle toy problem only teach Memorization? #card
Because the dataset is a single fixed, repeating loop of words. There is no "unseen" data to test the model with, so we only prove that it can memorize the loop.
