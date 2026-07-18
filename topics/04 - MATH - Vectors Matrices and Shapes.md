# Vectors, Matrices, and Shapes

## One-line definition
A Scalar is a single number, a Vector is a 1D list of numbers, and a Matrix is a 2D table of numbers; their Shapes dictate how they can interact.

## Why it exists
To organize data so that computer hardware (like GPUs) can perform thousands of calculations simultaneously.

## Beginner intuition
- **Scalar:** A single Excel cell.
- **Vector:** A single column (or row) in Excel.
- **Matrix:** A full spreadsheet table.

## Week 1 assignment connection
Our input `x` is a 1D Vector of shape `(10,)`. Our weights `W1` form a 2D Matrix of shape `(10, 8)`.

## Small numerical example
Scalar: `5`
Vector: `[1, 2, 3]` (Shape: `(3,)`)
Matrix: 
```
[[1, 2], 
 [3, 4], 
 [5, 6]]
```
(Shape: `(3, 2)` -> 3 rows, 2 columns)

## Common misunderstanding
**Misunderstanding:** NumPy automatically transposes 1D vectors to make matrix multiplication work.
**Correction:** A NumPy 1D array of shape `(n,)` is neither a row nor a column. When you use `np.dot()`, NumPy dynamically treats it as a row `(1, n)` or column `(n, 1)` depending on whether it is on the left or right side of the matrix, without actually changing its underlying shape.

## What happens if removed or changed?
If the inner shapes of a vector and matrix do not match, the program will throw a `ValueError: shapes not aligned`.

## Teach-back question
What is the difference between a shape of `(5,)`, `(1, 5)`, and `(5, 1)`?

## Flashcards

What is the exact difference between a Scalar, a Vector, and a Matrix? #card
A Scalar is a single number. A Vector is an ordered 1D list of numbers. A Matrix is a 2D rectangular table of numbers.

If a NumPy array has shape `(10,)`, is it a row matrix or a column matrix? #card
Neither. It is a 1D array. NumPy dynamically treats it as a row or column during `np.dot` depending on its position, but its true shape remains `(10,)`.
