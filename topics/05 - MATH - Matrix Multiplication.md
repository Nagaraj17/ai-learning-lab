# Matrix Multiplication and Dot Product

## One-line definition
Matrix multiplication is calculating the dot product of rows and columns to efficiently compute many weighted combinations simultaneously.

## Why it exists
If we had to calculate every single connection between neurons one-by-one using `for` loops, it would take years. Matrix multiplication does it instantly.

## Beginner intuition
Imagine you have 10 ingredients (vector `x`) and 8 different recipes (the columns of `W`). Matrix multiplication instantly calculates how much of each ingredient goes into all 8 recipes at the exact same time.

## Week 1 assignment connection
`np.dot(x, W1)` takes our one-hot vector (shape `(10,)`) and multiplies it against our weight matrix (shape `(10, 8)`), outputting a vector of shape `(8,)` representing the input to the 8 hidden neurons.

## Small numerical example
Vector `x` = `[1, 2]`
Matrix `W` = 
```
[[3, 4],
 [5, 6]]
```
`x @ W` = `[(1*3 + 2*5), (1*4 + 2*6)]` = `[13, 16]`

## Common misunderstanding
**Misunderstanding:** Matrix Multiplication is element-wise multiplication (multiplying the top-left with the top-left).
**Correction:** Matrix multiplication involves multiplying rows by columns and *summing* the results (dot product). In Python, `*` is element-wise, while `np.dot()` or `@` is matrix multiplication.

## What happens if removed or changed?
If we use `*` instead of `np.dot()`, the shapes will usually clash and cause an error, or it will just blindly multiply elements instead of combining them.

## Teach-back question
If a vector has shape `(10,)` and a matrix has shape `(10, 8)`, what is the shape of the result after matrix multiplication?

## Flashcards

What is the fundamental difference between `*` and `np.dot()` in NumPy? #card
`*` performs element-wise multiplication. `np.dot()` performs the dot product (matrix multiplication), combining rows and columns.

In the calculation `x @ W`, what does the matrix multiplication mathematically achieve for a neural network? #card
It efficiently calculates all the weighted combinations for every neuron in the next layer simultaneously, instead of using slow `for` loops.
