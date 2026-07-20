# Hidden Layers and tanh

> [!NOTE]
> This topic is based on Chapter 6 (Deep Feedforward Networks) of the *Deep Learning* textbook.

## Formal Definition
A linear model, such as linear regression, can only represent linear functions. To represent non-linear functions (which are necessary to solve complex real-world problems), we use deep feedforward networks with **hidden layers**.
The design of a hidden unit includes an **activation function** $g$ applied to an affine transformation:
$\mathbf{h} = g(\mathbf{W}^\top \mathbf{x} + \mathbf{b})$

One popular choice for $g$ is the hyperbolic tangent function, $\tanh$.
$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$

## Component-by-Component Math Breakdown
Let's break down the hidden layer formula: $\mathbf{h} = \tanh(\mathbf{z})$
- **$\mathbf{z}$**: This is the linear, raw output of the weights and biases we calculated in the previous step ($\mathbf{W}\mathbf{x} + \mathbf{b}$). It can be any number, from $-\infty$ to $\infty$.
- **$e^z$**: Euler's number ($2.718...$) raised to the power of our input $z$. This makes large positive numbers explode and large negative numbers shrink to $0$.
- **$\tanh(z)$**: This function takes any number $z$ and mathematically forces (squashes) it to be strictly between $-1$ and $1$.
- **$\mathbf{h}$**: The final "activation" of the hidden layer. This is the output that gets passed to the next layer.

## Beginner Intuition & Contrasting Analogy
Why do we need an activation function at all? 

Imagine you are trying to separate a group of red dots from a group of blue dots drawn on a piece of paper.
- **Without a hidden layer/activation:** You are only allowed to draw a **straight line** to separate them. If the blue dots form a circle around the red dots, a straight line will *never* work. The model fails.
- **With a hidden layer/activation:** It is like you are allowed to **fold the paper** (warp the space) before drawing your line! The $\tanh$ function acts as the "folder", bending the straight line into an 'S' shape. By combining hundreds of these 'S' shapes, the AI can fold the paper around complex boundaries, drawing a circle to perfectly isolate the red dots.

![tanh Activation Function](<images/tanh_activation.png>)

## Where is this used in AI?
*   **Universal Approximation:** Because of activation functions like $\tanh$, a Neural Network with just one hidden layer is mathematically proven to be a "Universal Approximator." It can learn to represent *any* mathematical function in the universe, no matter how complex!
*   **Recurrent Neural Networks (RNNs):** While modern networks often use ReLU (which we cover in an optional topic), $\tanh$ is still heavily used in models dealing with time and sequence, like LSTMs, because its strict $[-1, 1]$ boundaries keep the network's internal state from exploding to infinity over long sequences.

## Small Numerical Example
Let's see how $\tanh$ squashes inputs:
- Linear Output $\mathbf{z} = 0.0 \rightarrow \tanh(0.0) = 0.0$
- Linear Output $\mathbf{z} = 3.5 \rightarrow \tanh(3.5) \approx 0.998$
- Linear Output $\mathbf{z} = -5.0 \rightarrow \tanh(-5.0) \approx -0.999$

Even though $3.5$ and $100$ are very different, their $\tanh$ values are both basically $1$. $\tanh$ creates a strong cutoff.

*(Source: Ian Goodfellow, Yoshua Bengio, and Aaron Courville - Deep Learning, Chapter 6)*

---

## Flashcards

What happens mathematically if you remove the activation function from a Neural Network? #card
The entire network collapses into a single linear equation. No matter how many hidden layers you have, a straight line times a straight line is still just a straight line. The model loses its ability to learn complex, non-linear shapes.

What is the output range of the `tanh` activation function? #card
It bounds the output strictly between `-1` and `1`, and is zero-centered. Inputs around +3 or -3 already push the output very close to the extreme bounds.
