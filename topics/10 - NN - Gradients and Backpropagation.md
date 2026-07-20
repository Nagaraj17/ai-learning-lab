# Gradients and Backpropagation

> [!NOTE]
> This topic is based on Chapter 6.5 (Back-Propagation) of the *Deep Learning* textbook.

## Formal Definition
When training a neural network, we need to know how changing any given weight will affect the final loss. We calculate this using a **gradient**, which is the vector of partial derivatives of the loss with respect to the parameters. **Back-propagation** is the algorithm used to efficiently compute these gradients by applying the chain rule of calculus backwards from the output layer to the input layer.

## Component-by-Component Math Breakdown
The core of backpropagation is the calculus **chain rule**. If we have a sequence of operations like a weight $\mathbf{W}$ creating a hidden state $\mathbf{h}$, which then creates the final Loss $\mathbf{L}$:

$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial h} \cdot \frac{\partial h}{\partial W}$

- **$\partial$ (Partial Derivative)**: A mathematical way of asking "How much does a tiny change in the bottom variable affect the top variable?"
- **$\frac{\partial L}{\partial h}$**: "How much did the Hidden State affect the final Loss?" We calculate this first at the end of the network.
- **$\frac{\partial h}{\partial W}$**: "How much did the Weight affect the Hidden State?" We calculate this moving backwards.
- **$\cdot$ (Multiplication)**: We literally just multiply the two answers together to get the final gradient: $\frac{\partial L}{\partial W}$ (How much the Weight affected the Loss).

## Beginner Intuition & Contrasting Analogy
Imagine a massive assembly line building a car. 
The final inspector finds that the car door is deeply scratched (This is the **Loss**). 

**Backpropagation** is the inspector walking backwards down the assembly line to assign blame (Gradients). 
He first asks the Door-Installer ($\mathbf{h}$): "Did you scratch this?" The installer says, "I installed it, but it was already scratched when the Painter handed it to me."
The inspector multiplies this blame backwards to the Painter ($\mathbf{W}$): "Did you scratch this?" The painter says, "Ah, yes, my machine malfunctioned."

The algorithm *propagates* the error *backwards* through every worker, calculating exactly how much each person contributed to the scratch, so they know who needs to fix their machine.

![Backpropagation Chain Rule](<images/backpropagation_chain_rule.png>)

## Where is this used in AI?
*   **PyTorch Autograd:** This exact algorithm is the foundational magic of PyTorch and TensorFlow. When you call `loss.backward()` in PyTorch, the framework automatically walks backward through your entire code, applying the chain rule to every single mathematical operation, calculating the gradient for billions of weights in milliseconds. It is called "Differentiable Programming."
*   **Hardware Acceleration:** Backpropagation is mathematically just a massive chain of matrix multiplications. This is the exact reason GPUs (NVIDIA chips) are used for AI: they are incredibly fast at multiplying matrices in parallel.

## Small Numerical Example
Let's see how a Gradient tells us which direction to step:
- **Gradient is $+2.5$:** A positive slope. Increasing this weight will *increase* the loss. Because we want to *reduce* loss, we should decrease this weight!
- **Gradient is $-0.5$:** A negative slope. Increasing this weight will *decrease* the loss. Because we want to *reduce* loss, we should increase this weight!

**Rule of Thumb:** We always step in the *opposite* direction of the gradient to reach the bottom of the error valley.

*(Source: Ian Goodfellow, Yoshua Bengio, and Aaron Courville - Deep Learning, Chapter 6.5)*

---

## Flashcards

Does Backpropagation actually change the weights of the neural network? #card
No. Backpropagation purely calculates the Gradients (how much each parameter is to blame for the loss). The weights are changed in an entirely separate step using an Optimizer (like Gradient Descent).

What does the sign (positive or negative) of a Gradient tell us? #card
The direction of the slope. A positive gradient means increasing the weight increases the loss. A negative gradient means increasing the weight decreases the loss. We always update the weight in the opposite direction of the gradient.
