# Machine Learning Foundations

## One-line definition
Machine learning is teaching a system to recognize patterns in data (Supervised Learning uses labeled data), using a Dataset of Inputs and Targets.

## Why it exists
To solve problems that are too complex for hard-coded "if-then" rules. 

## Beginner intuition
Instead of telling the computer exactly how to solve a maze, you let it run through the maze thousands of times, giving it a reward when it finds the exit. Over time, it learns the path.

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
