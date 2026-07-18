# Learning Rate and Updates

## One-line definition
The **Weight Update** applies the gradients to fix the parameters, and the **Learning Rate** controls exactly how big of a step we take.

## Why it exists
To actually improve the model! Without the update step, calculating the gradients is useless.

## Beginner intuition
Imagine you are walking down a mountain blindfolded to find the lowest valley (Loss = 0). The Gradient tells you which way is downhill. The Learning Rate is the size of your footstep. 

## Week 1 assignment connection
In the `update_weights` function, we perform `W1 -= lr * dW1`. We use subtraction to ensure we always step *against* the slope (downhill). Our learning rate is `0.1`.

## Small numerical example
Formula: `new_weight = old_weight - (learning_rate * gradient)`
- Old weight: `0.5`
- Gradient: `2.0` (Positive slope)
- Learning Rate: `0.1`
- New weight = `0.5 - (0.1 * 2.0)` = `0.5 - 0.2` = **`0.3`**

## Common misunderstanding
**Misunderstanding:** The gradient is the weight update.
**Correction:** The gradient is just the slope. The actual update requires multiplying the slope by the learning rate.

## What happens if removed or changed?
- If Learning Rate is `0.0`: `new_weight = old_weight`. The network never learns.
- If Learning Rate is massive (e.g., `100.0`): The network takes a massive step, jumps across the valley to the other mountain, and the loss explodes to infinity.

## Teach-back question
Why do we *subtract* the gradient during the update step (`W1 = W1 - lr * dW1`), rather than adding it?

## Flashcards

What happens if you set the Learning Rate to 0.0? #card
The parameters never update (`old_weight - 0 * gradient = old_weight`). The network does not learn.

Why do we use subtraction in the weight update formula (`W -= lr * gradient`)? #card
Because the gradient points uphill (towards maximum loss). We want to minimize the loss, so we subtract the gradient to step downhill.
