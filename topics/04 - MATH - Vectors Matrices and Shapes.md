# Vectors, Matrices, and Shapes

## One-line definition
A Scalar is a single number, a Vector is an array of numbers, and a Matrix is a 2D table of numbers; their Shapes dictate how they can interact.

## Formal Definition (from *Deep Learning*, Chapter 2.1)
*   **Scalars:** A scalar is just a single number, in contrast to most of the other objects studied in linear algebra, which are usually arrays of multiple numbers. We write scalars in italics, e.g., $s \in \mathbb{R}$.
*   **Vectors:** A vector is an array of numbers. The numbers are arranged in order. We can identify each individual number by its index in that ordering. We write vectors in bold lowercase, e.g., $\mathbf{x} \in \mathbb{R}^n$.
*   **Matrices:** A matrix is a 2D array of numbers, so each element is identified by two indices instead of just one. We write matrices in bold uppercase, e.g., $\mathbf{A} \in \mathbb{R}^{m \times n}$.

*(Source: Ian Goodfellow, Yoshua Bengio, and Aaron Courville - Deep Learning, Chapter 2.1)*

## Beginner intuition
While the formal math uses symbols like $\mathbb{R}^{m \times n}$, we can think of it in terms of a spreadsheet:
- **Scalar:** A single Excel cell.
- **Vector:** A single column (or row) in Excel.
- **Matrix:** A full spreadsheet table.

![Vector Dimensionality vs NumPy Arrays](<../weekly curriculum/01-forward-pass-and-learning/images/vector_dimensionality.png>)

## Why it exists
To organize data so that computer hardware (like GPUs) can perform thousands of calculations simultaneously using linear algebra rules, rather than doing them one by one.

## Week 1 assignment connection
Our input $\mathbf{x}$ is a 1D Vector of shape `(10,)`. Our weights $\mathbf{W}$ form a 2D Matrix of shape `(10, 8)`.

## Small numerical example
Scalar $s = 5$

Vector $\mathbf{x} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ (Shape: `(3,)`)

Matrix $\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}$ (Shape: `(3, 2)` -> 3 rows, 2 columns)

![Scalar, Vector, and Matrix Shapes](<../weekly curriculum/01-forward-pass-and-learning/images/matrix_shapes.png>)

## Common misunderstanding
**Misunderstanding:** NumPy automatically transposes 1D vectors to make matrix multiplication work.
**Correction:** A NumPy 1D array of shape `(n,)` is neither a row nor a column. When you use `np.dot()`, NumPy dynamically treats it as a row `(1, n)` or column `(n, 1)` depending on whether it is on the left or right side of the matrix, without actually changing its underlying shape.

## What happens if removed or changed?
If the inner shapes of a vector and matrix do not match, the program will throw a `ValueError: shapes not aligned`.

## Teach-back question
What is the difference between a shape of `(5,)`, `(1, 5)`, and `(5, 1)`?

## Flashcards

What is the exact difference between a Scalar, a Vector, and a Matrix? #card
A Scalar is a single number ($x \in \mathbb{R}$). A Vector is an ordered 1D array of numbers ($\mathbf{x} \in \mathbb{R}^n$). A Matrix is a 2D rectangular table of numbers ($\mathbf{A} \in \mathbb{R}^{m \times n}$).

If a NumPy array has shape `(10,)`, is it a row matrix or a column matrix? #card
Neither. It is a 1D array. NumPy dynamically treats it as a row or column during `np.dot` depending on its position, but its true shape remains `(10,)`.
