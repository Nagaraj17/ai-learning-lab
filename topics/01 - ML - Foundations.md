# Machine Learning Foundations

## One-line definition
Machine learning is teaching a system to recognize patterns in data (Supervised Learning uses labeled data), using a Dataset of Inputs and Targets.

## Why it exists
To solve problems that are too complex for hard-coded "if-then" rules.

![Traditional Programming vs Machine Learning — rule-based vs data-driven](<../weekly curriculum/01-forward-pass-and-learning/images/traditional_vs_ml.png>)

*Traditional programming gives the computer rules and data to produce answers. Machine learning gives it data and answers so it can discover the rules itself.*

## Beginner intuition
Think of a student learning from a practice exam that comes with an answer key. The student makes a guess for each question, checks it against the answer key, and uses the corrections to improve. Over many rounds, the student learns the patterns. Machine learning works exactly the same way — examples are the questions, labels are the answer key, and the network adjusts its internal numbers after each correction.

## Week 1 assignment connection
In `01_Next_Word_Predictor.ipynb`, our Input is a current lifecycle state (e.g., "Order") and our Target is the expected next state ("Shipment"). We are using Supervised Learning to train the network on this specific Dataset.

## Small numerical example
Input: `x = [1, 0]`
Target (Label): `y = [0, 1]`
The model predicts `[0.6, 0.4]`. The error (loss) tells it how far off it was from the Target `[0, 1]`.

## Common misunderstanding
**Misunderstanding:** The model "understands" the words. 
**Correction:** The model only understands mathematical relationships between numbers; it has no concept of what an "Order" actually is in the real world.

## What happens if removed or changed?
If we remove the Targets/Labels, we can no longer do Supervised Learning. The model wouldn't know when it made a mistake, so it could never correct its weights.

## Teach-back question
Why can't we just write an `if` statement for predicting the next word in the PO Lifecycle? (And why is this a toy problem where an `if` statement actually *would* work better?)

## Flashcards

What is the difference between Traditional Programming and Machine Learning? #card
Traditional Programming requires a human to write explicit rules to process data into answers. Machine Learning takes data and answers, and learns the rules itself.

In Supervised Learning, what do we call the "correct answer" we provide to the model during training? #card
The Target or Label.
