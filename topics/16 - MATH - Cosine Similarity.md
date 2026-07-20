# Cosine Similarity

> [!NOTE]
> This topic covers the mathematical formula used to prove that a neural network has actually learned the semantic meaning of words.

## Formal Definition
If two words mean the exact same thing, their embedding vectors should point to the exact same location in space.
**Cosine Similarity** is a mathematical metric used to measure how similar two vectors are by looking exclusively at the *angle* between them, ignoring their magnitude (length). 

The formal equation for Cosine Similarity between vector $\mathbf{A}$ and vector $\mathbf{B}$ is:
$$ \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{||\mathbf{A}|| \times ||\mathbf{B}||} $$

## Component-by-Component Math Breakdown
- **$\mathbf{A} \cdot \mathbf{B}$**: The mathematical Dot Product of the two vectors. (Multiply matching elements and sum them up).
- **$||\mathbf{A}||$**: The $L^2$ Norm (Magnitude) of vector $\mathbf{A}$. This is the physical length of the vector, calculated using the Pythagorean theorem.
- **$\frac{\dots}{||\mathbf{A}|| \times ||\mathbf{B}||}$**: By dividing the dot product by the absolute lengths of the vectors, we mathematically "cancel out" their lengths. This isolates only the angle $\theta$ between them.
- **$\cos(\theta)$**: The cosine of the angle between them. The result is always a number strictly between `-1.0` and `1.0`.

## Beginner Intuition & Contrasting Analogy
Imagine you and your friend are both pointing at the exact same star in the night sky. 
- Your friend is short, so they extend their arm 2 feet. 
- You are tall, so you extend your arm 3 feet. 
If we measure the physical distance between your fingertips (Euclidean Distance), you are 1 foot apart. But if we measure the *angle* of your arms, the angle is 0 degrees. You are pointing in the exact same direction.

In NLP, common words like "The" get trained a lot, so their vector arms grow very long. Rare words have short arms. Cosine similarity ignores the length of the arm and only looks at where it's pointing!

```mermaid
graph TD
    subgraph Perfect Similarity = 1.0
    A1["Vector A (Length 3)"] --> C1["Same Direction"]
    B1["Vector B (Length 5)"] --> C1
    end
    
    subgraph Zero Similarity = 0.0
    A2["Vector A"] --> C2["90 Degrees Apart (Orthogonal)"]
    B2["Vector B"] --> C2
    end
    
    subgraph Opposite Similarity = -1.0
    A3["Vector A"] --> C3["180 Degrees Apart (Opposite)"]
    B3["Vector B"] --> C3
    end
    
    style Perfect Similarity fill:#eef9ff,stroke:#333
    style Zero Similarity fill:#f9f9f9,stroke:#333
    style Opposite Similarity fill:#ffeef0,stroke:#333
```

## Where is this used in AI?
*   **Vector Databases (RAG):** When you upload a massive PDF document to ChatGPT and ask it a question, ChatGPT doesn't read the whole PDF. It converts your question into an embedding vector, and then calculates the **Cosine Similarity** against every single paragraph in the PDF. It grabs the paragraph with the highest Cosine score (e.g. `0.95`) and uses that text to answer your question!
*   **Checking our Work:** In our assignment, to prove that our neural network actually learned the lifecycle, we will extract the embedding for `"Receive"` and `"Restock"` and calculate their Cosine Similarity. We expect it to be a high positive number (e.g. `0.85`), proving they clustered together in space.

## Small Numerical Example
Vector $\mathbf{A}$: `[3, 4]` (Magnitude = 5)
Vector $\mathbf{B}$: `[6, 8]` (Magnitude = 10)
- Dot Product: $(3 \times 6) + (4 \times 8) = 18 + 32 = 50$
- Product of Magnitudes: $5 \times 10 = 50$
- Cosine Similarity: $50 / 50 = \mathbf{1.0}$ (They point in the exact same direction!).

## Common Misunderstanding
**Misunderstanding:** A Cosine Similarity of `0.0` means the words are opposites.
**Correction:** A similarity of `0.0` means they are completely unrelated (Orthogonal / 90 degrees). For example, "Apple" and "Carburetor". A similarity of `-1.0` means they are exact opposites (180 degrees). For example, "Hot" and "Cold".

---

## Flashcards

Why is Cosine Similarity preferred over Euclidean distance (physical distance) for comparing word embeddings? #card
Because embeddings for common words can grow longer (higher magnitude) than rare words. Cosine similarity divides by the magnitude to cancel it out. It only measures the angle (direction) between vectors, which is a much purer representation of semantic meaning.

What does a Cosine Similarity of 1.0, 0.0, and -1.0 mean respectively? #card
1.0 means identical direction (high semantic similarity). 0.0 means orthogonal (completely unrelated). -1.0 means opposite direction (exact opposites).
