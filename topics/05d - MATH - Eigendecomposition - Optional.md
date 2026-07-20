# Eigendecomposition (Optional)

> [!NOTE]
> This is an optional deep-dive topic based on Chapter 2.7 of the *Deep Learning* textbook. It provides the foundation for advanced concepts like Principal Component Analysis (PCA).

## Formal Definition (Chapter 2.7)
Many mathematical objects can be understood better by breaking them into constituent parts, or finding some of their properties that are universal, not caused by the way we choose to represent them. We can decompose matrices in ways that show us information about their functional properties that is not obvious from the representation of the matrix as an array of elements.

One of the most widely used kinds of matrix decomposition is called **eigendecomposition**, in which we decompose a matrix into a set of eigenvectors and eigenvalues.
An eigenvector of a square matrix $\mathbf{A}$ is a non-zero vector $\mathbf{v}$ such that multiplication by $\mathbf{A}$ alters only the scale of $\mathbf{v}$:
$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$
The scalar $\lambda$ is known as the eigenvalue corresponding to this eigenvector.

## Beginner Intuition
When you multiply a vector by a matrix, it usually changes the vector's direction and stretches it. 
However, for any given matrix, there are special "magic" vectors called **eigenvectors**. When the matrix multiplies its own eigenvectors, it doesn't change their direction at all! It only stretches or shrinks them. 
The amount of stretching or shrinking is called the **eigenvalue**.

This is incredibly useful because it tells us the "primary directions" or "axes" along which a matrix transformation operates.

*(Source: Ian Goodfellow, Yoshua Bengio, and Aaron Courville - Deep Learning, Chapter 2.7)*
