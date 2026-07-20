# Identity, Inverse, and Special Matrices (Optional)

> [!NOTE]
> This is an optional deep-dive topic based on Chapter 2.3 & 2.6 of the *Deep Learning* textbook. 

## Identity Matrix (Chapter 2.3)

### Formal Definition
An identity matrix is a matrix that does not change any vector when we multiply that vector by that matrix. We denote the identity matrix that preserves $n$-dimensional vectors as $I_n$. Formally, $I_n \in \mathbb{R}^{n \times n}$, and
$\forall \mathbf{x} \in \mathbb{R}^n, I_n \mathbf{x} = \mathbf{x}$

### Beginner Intuition
The Identity Matrix is the matrix equivalent of the number **1**. Just like $5 \times 1 = 5$, multiplying any matrix by an identity matrix returns the original matrix. It is a square matrix with $1$s on the main diagonal and $0$s everywhere else.

## Matrix Inverse (Chapter 2.3)

### Formal Definition
The matrix inverse of $\mathbf{A}$ is denoted as $\mathbf{A}^{-1}$, and it is defined as the matrix such that:
$\mathbf{A}^{-1} \mathbf{A} = I_n$

### Beginner Intuition
If the Identity Matrix is like the number $1$, then the Matrix Inverse is like division (e.g. multiplying by $1/x$). It "undoes" the transformation applied by the original matrix $\mathbf{A}$.

## Orthogonal Matrices (Chapter 2.6)

### Formal Definition
An orthogonal matrix is a square matrix whose rows are mutually orthonormal and whose columns are mutually orthonormal:
$\mathbf{A}^T \mathbf{A} = \mathbf{A} \mathbf{A}^T = I$
This implies that $\mathbf{A}^{-1} = \mathbf{A}^T$.

### Beginner Intuition
Orthogonal matrices represent transformations that preserve lengths and angles, like rotating or flipping a shape without stretching or squishing it. They are computationally very cheap to invert, because you just transpose them!

*(Source: Ian Goodfellow, Yoshua Bengio, and Aaron Courville - Deep Learning, Chapter 2.3 & 2.6)*
