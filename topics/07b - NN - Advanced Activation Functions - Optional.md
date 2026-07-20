# Advanced Activation Functions (Optional)

> [!NOTE]
> This topic is based on Chapter 6.3 (Hidden Units) of the *Deep Learning* textbook.

## Formal Definition
While `tanh` is historically significant and useful for recurrent networks, modern feedforward and convolutional neural networks almost exclusively use **Rectified Linear Units (ReLU)**. The ReLU activation function is remarkably simple, yet it prevents the "vanishing gradient" problem that plagues `tanh` and `sigmoid` functions in very deep networks.

Formally, ReLU is defined as:
$g(z) = \max(0, z)$

## Component-by-Component Math Breakdown
- **$z$**: The raw, linear pre-activation output from the weights and biases ($z = \mathbf{W}\mathbf{x} + \mathbf{b}$).
- **$\max(0, z)$**: This is a programming logic gate. 
  - If $z$ is negative (e.g., $-5$), the function returns $0$. 
  - If $z$ is positive (e.g., $+12$), the function returns the exact same number ($+12$).
- **$g(z)$**: The final activated output passed to the next layer.

## Beginner Intuition & Contrasting Analogy
Imagine a network of water pipes with automatic shutoff valves.
- **`tanh` (The strict regulator):** A valve that takes incoming water pressure and rigidly caps it. No matter how much pressure comes in, a maximum of 1 gallon/sec goes out. If the pressure is too high, it gets squashed.
- **ReLU (The smart shutoff):** A valve that only stops backwards flow. If the water flows backward (negative), it slams shut and lets $0$ water through. If the water flows forward (positive), it just opens up completely and lets *all* the pressure through, untouched!

This property is magical because by letting positive signals flow through untouched, it prevents the training signals (gradients) from dying out as they travel backward through hundreds of layers!

![ReLU Activation Function](<images/relu_activation_function.png>)

## Where is this used in AI?
*   **Computer Vision (CNNs):** Almost every modern image recognition model (like ResNet or YOLO) uses ReLU as its default hidden layer activation.
*   **"Dying ReLU" Problem:** Because ReLU strictly outputs $0$ for negative numbers, sometimes a neuron's weights get pushed so negative that it *never* activates again (it "dies"). To fix this, AI engineers invented **Leaky ReLU**, which lets a tiny trickle of negative numbers through: $g(z) = \max(0.01z, z)$.

## Small Numerical Example
Let's see how ReLU processes a vector of linear outputs:
- **Linear Output $\mathbf{z}$:** $[3.5, -1.2, 0.0, 100.5, -42.1]$
- **ReLU Applied:** $[3.5, 0.0, 0.0, 100.5, 0.0]$

Notice how the large positive number $100.5$ wasn't squashed down to $1$ (like it would be with `tanh`). It was preserved perfectly.

*(Source: Ian Goodfellow, Yoshua Bengio, and Aaron Courville - Deep Learning, Chapter 6.3)*

---

## Flashcards

What is the mathematical formula for the ReLU activation function? #card
$g(z) = \max(0, z)$. If the input is negative, it returns 0. If the input is positive, it returns the input unchanged.

Why is ReLU often preferred over `tanh` in deep networks? #card
Because `tanh` squashes large positive and negative numbers, causing their gradients to become near-zero (the Vanishing Gradient Problem). ReLU does not squash positive numbers, allowing gradients to flow backward powerfully through hundreds of layers.
