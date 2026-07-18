# Level 2: Cosine Similarity

This exercise traces the calculation of Cosine Similarity between two embedding vectors.

## Initialization
We have extracted two embedding vectors from our trained neural network.
- **Vector A ("Receive"):** `[3, 4]`
- **Vector B ("Restock"):** `[6, 8]`

---

## 1. Calculate the Dot Product ($A \cdot B$)
Multiply matching elements and sum them up.

- $(3 \times 6) + (4 \times 8)$
- $18 + 32$
- **Result:** $50$

## 2. Calculate the Magnitude of Vector A ($||A||$)
Square each element, sum them, and take the square root (Pythagorean theorem).

- $\sqrt{3^2 + 4^2}$
- $\sqrt{9 + 16}$
- $\sqrt{25}$
- **Result:** $5$

## 3. Calculate the Magnitude of Vector B ($||B||$)
- $\sqrt{6^2 + 8^2}$
- $\sqrt{36 + 64}$
- $\sqrt{100}$
- **Result:** $10$

## 4. Calculate Cosine Similarity
Divide the Dot Product by the product of the Magnitudes.

- $\text{Cosine Sim} = \frac{A \cdot B}{||A|| \times ||B||}$
- $\text{Cosine Sim} = \frac{50}{5 \times 10}$
- $\text{Cosine Sim} = \frac{50}{50}$
- **Result:** $1.0$ (Perfect Similarity!)

*Notice that even though Vector B is twice as long as Vector A, their Cosine Similarity is 1.0 because they point in the exact same direction.*

---

## Learner Workspace

### Predict the Sign
If Vector C is `[-3, -4]` and Vector A is `[3, 4]`, they are pointing in exactly opposite directions. What will their Cosine Similarity be?

### Blank Calculation Table
Try calculating the Cosine Similarity between these two orthogonal (unrelated) vectors:
- Vector X: `[0, 5]`
- Vector Y: `[5, 0]`

| Step | Formula | Your Calculation | Expected Output |
|---|---|---|---|
| Dot Product | $X \cdot Y$ | | |
| Magnitude X | $||X||$ | | |
| Magnitude Y | $||Y||$ | | |
| Cosine Sim | $\frac{X \cdot Y}{||X|| \times ||Y||}$ | | |
