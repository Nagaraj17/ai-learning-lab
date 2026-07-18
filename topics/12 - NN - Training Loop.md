# Training Loop and Epoch

## One-line definition
An **Epoch** is one complete pass of the entire dataset, and the **Training Loop** is the repetitive cycle of guessing, calculating error, and updating weights.

## Why it exists
A neural network cannot learn everything in a single step. It must iteratively inch down the error mountain by seeing the data thousands of times.

## Beginner intuition
It's like practicing a piano scale. 
One scale = One data pair. 
Playing all 10 scales in the book once = One Epoch. 
You have to loop through the book 500 times (500 Epochs) before your fingers memorize the movements perfectly.

## Week 1 assignment connection
We loop `for epoch in range(500):` and inside that, we loop through our `training_data`. The inner loop executes the 4 core steps: `forward_pass`, `compute_loss`, `backward_pass`, and `update_weights`.

## Small numerical example
If our dataset has 6 pairs, and we train for 500 epochs:
The `update_weights` function executes `6 * 500 = 3,000` times.

## Common misunderstanding
**Misunderstanding:** One epoch means the network updated its weights once.
**Correction:** In our implementation, weights update after *every single example*. So one epoch actually contains 6 weight updates (because there are 6 examples in the dataset).

## What happens if removed or changed?
If we only run 1 epoch, the network takes 6 tiny steps downhill. The loss might drop from `2.3` to `2.2`. It will still be mostly wrong. Training requires patience and iteration.

## Teach-back question
If you have 10 examples in your dataset, and you train for 10 epochs, exactly how many times did the Forward Pass function run?

## Flashcards

What is the definition of an Epoch? #card
One Epoch is one complete pass of the entire training dataset through the neural network.

List the 4 core steps executed inside a Training Loop for a single example. #card
1. Forward Pass (Make a prediction)
2. Loss Calculation (Measure the error)
3. Backward Pass (Calculate gradients)
4. Parameter Update (Adjust weights)
