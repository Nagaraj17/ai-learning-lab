# Matrix Multiplication and Dot Product

## One-line definition
Matrix multiplication is calculating the dot product of rows and columns to efficiently compute many weighted combinations simultaneously.

## Formal Definition (from *Deep Learning*, Chapter 2.2)
One of the most important operations involving matrices is multiplication of two matrices. The matrix product of matrices $\mathbf{A}$ and $\mathbf{B}$ is a third matrix $\mathbf{C}$. In order for this product to be defined, $\mathbf{A}$ must have the same number of columns as $\mathbf{B}$ has rows. If $\mathbf{A}$ is of shape $m \times n$ and $\mathbf{B}$ is of shape $n \times p$, then $\mathbf{C}$ is of shape $m \times p$.
The product operation is defined by:
$C_{i,j} = \sum_{k} A_{i,k} B_{k,j}$

*(Source: Ian Goodfellow, Yoshua Bengio, and Aaron Courville - Deep Learning, Chapter 2.2)*

## Beginner intuition
That sum formula $\sum$ just means: take a row of the first matrix, match it up with a column of the second matrix, multiply the matching pairs, and add them all up. 
Imagine you have 10 ingredients (vector $\mathbf{x}$) and 8 different recipes (the columns of $\mathbf{W}$). Matrix multiplication instantly calculates how much of each ingredient goes into all 8 recipes at the exact same time.

![Matrix Multiplication Visualized](<../weekly curriculum/01-forward-pass-and-learning/images/matrix_multiplication.png>)

## Why it exists
If we had to calculate every single connection between neurons one-by-one using `for` loops, it would take years. Matrix multiplication does it instantly, leveraging highly optimized hardware (like GPUs) built exactly for this operation.

## Week 1 assignment connection
`np.dot(x, W1)` takes our one-hot vector (shape `(10,)`) and multiplies it against our weight matrix (shape `(10, 8)`), outputting a vector of shape `(8,)` representing the input to the 8 hidden neurons.

## Small numerical example
Vector $\mathbf{x} = \begin{bmatrix} 1 & 2 \end{bmatrix}$
Matrix $\mathbf{W} = \begin{bmatrix} 3 & 4 \\ 5 & 6 \end{bmatrix}$

$\mathbf{x} \mathbf{W} = \begin{bmatrix} (1 \cdot 3 + 2 \cdot 5) & (1 \cdot 4 + 2 \cdot 6) \end{bmatrix} = \begin{bmatrix} 13 & 16 \end{bmatrix}$

## Common misunderstanding
**Misunderstanding:** Matrix Multiplication is element-wise multiplication (multiplying the top-left with the top-left).
**Correction:** Matrix multiplication involves multiplying rows by columns and *summing* the results (dot product), just like the formula $C_{i,j} = \sum_k A_{i,k} B_{k,j}$ shows. In Python, `*` is element-wise, while `np.dot()` or `@` is matrix multiplication.

## What happens if removed or changed?
If we use `*` instead of `np.dot()`, the shapes will usually clash and cause an error, or it will blindly multiply elements instead of properly combining them via the dot product.

## Teach-back question
If a vector has shape `(10,)` and a matrix has shape `(10, 8)`, what is the shape of the result after matrix multiplication?

## Flashcards

What is the fundamental difference between `*` and `np.dot()` in NumPy? #card
`*` performs element-wise multiplication. `np.dot()` performs the dot product (matrix multiplication), combining rows and columns according to $C_{i,j} = \sum_k A_{i,k} B_{k,j}$.

In the calculation $\mathbf{x} \mathbf{W}$, what does the matrix multiplication mathematically achieve for a neural network? #card
It efficiently calculates all the weighted combinations for every neuron in the next layer simultaneously, instead of using slow `for` loops.
