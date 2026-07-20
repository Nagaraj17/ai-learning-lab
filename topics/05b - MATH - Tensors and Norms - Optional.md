# Tensors and Norms (Optional)

> [!NOTE]
> This is an optional deep-dive topic based on Chapter 2.1 & 2.5 of the *Deep Learning* textbook. You do not need to master this to complete the Forward Pass assignment, but it is highly useful for advanced AI architectures.

## Tensors (Chapter 2.1)

### Formal Definition
In some cases we will need an array with more than two axes. In the general case, an array of numbers arranged on a regular grid with a variable number of axes is known as a tensor. We denote a tensor with this typeface: $\textsf{A}$. 

### Beginner Intuition
If a scalar is a point (0D), a vector is a line (1D), and a matrix is a flat sheet (2D), then a tensor is a 3D cube (or a 4D hypercube, etc.). 

### Where is this used in AI?
*   **Images (3D Tensors):** A color image isn't just a flat grid of pixels; it has 3 color channels (Red, Green, Blue). So an image is represented as a 3D Tensor of shape `(Height, Width, 3)`.
*   **Batches (4D Tensors):** In deep learning, we don't train on one image at a time. We train on a "batch" of images (e.g., 32 images). That creates a 4D Tensor of shape `(32, Height, Width, 3)`. Tensors allow the GPU to process all 32 images simultaneously.

---

## Norms (Chapter 2.5)

### Formal Definition
Sometimes we need to measure the size of a vector. In machine learning, we usually measure the size of vectors using a function called a **norm**. Formally, the $L^p$ norm is given by:
$||\mathbf{x}||_p = (\sum_i |x_i|^p)^{1/p}$

The $L^2$ norm, with $p=2$, is known as the Euclidean norm.

### Beginner Intuition
A norm is just a mathematical way to calculate the "length" or "magnitude" of a vector. But how you measure length depends on the rules of the world you are in!

Imagine you are trying to get from your house to a grocery store that is 3 miles East and 4 miles North.

- **The $L^2$ norm (Euclidean norm)** is how a bird would measure the distance. A bird can fly over buildings in a perfectly straight diagonal line. Using the Pythagorean theorem ($3^2 + 4^2 = 5^2$), the bird flies exactly 5 miles.
- **The $L^1$ norm (Manhattan distance)** is how a human walking on grid-like city blocks would measure the distance. You can't walk diagonally through buildings! You must walk 3 miles East along the street, and then 4 miles North. The total distance you walked is $3 + 4 = 7$ miles.

![L1 vs L2 Norm Visualization](<images/norms_visualization.png>)

### Where is this used in AI?
Norms are crucial because AI models need a mathematical way to measure "how wrong" they are, or "how big" their weights are getting.
*   **Loss Functions (Measuring Error):** 
    Let's say your AI makes two predictions. It gets the first one wrong by $10, and the second one wrong by $1,000. We can think of the total error as a vector: $\mathbf{e} = [10, 1000]$. We need a single number to represent "how bad" the AI is doing so we can train it to do better.
    - If we use the **$L^1$ Norm** (Mean Absolute Error), we just add them up: $10 + 1000 = 1010$. Every error is treated equally.
    - If we use the **$L^2$ Norm** (Mean Squared Error), we square the errors before adding them: $10^2 + 1000^2 = 100 + 1,000,000 = 1,000,100$. 
    
    **Why this matters:** Notice how the $L^2$ norm absolutely exploded because of the $1,000 error? The $L^2$ norm **heavily penalizes large mistakes**. If you use $L^2$ loss, your AI will desperately try to avoid making huge errors, even if it means making a bunch of tiny errors instead.

    ![L1 vs L2 Loss Penalty](<images/loss_function_norms.png>)

*   **Regularization (Preventing Overfitting):** If a neural network's weights (the numbers in its matrices) get too large, it means the model is "memorizing" the data rather than learning patterns. We calculate the **$L^2$ Norm** of the weights and penalize the model if the "length" of the weight vector gets too big! This is called "Weight Decay".
*   **Embeddings (Word Similarities):** In Large Language Models, words are converted into vectors. To see if "King" and "Queen" are similar, we calculate the $L^2$ distance between their vectors!

*(Source: Ian Goodfellow, Yoshua Bengio, and Aaron Courville - Deep Learning, Chapter 2.1 & 2.5)*
