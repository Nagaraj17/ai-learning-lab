# Cosine Similarity

## One-line definition
**Cosine Similarity** is a metric used to measure how similar two vectors are by looking at the *angle* between them, rather than the distance between their points.

## Why it exists
In high-dimensional embedding spaces, vectors can grow in magnitude (length) depending on how frequently words appeared in training. If we just measured the physical distance between two points (Euclidean distance), a rare word and a common word might look far apart, even if they mean the exact same thing. By measuring the angle, we focus purely on the *direction* of the meaning.

## Beginner intuition
Imagine you and your friend are both pointing at the same star in the night sky. Your friend extends their arm 2 feet. You extend your arm 3 feet. 
If we measure the distance between your fingertips, you are 1 foot apart. But if we measure the *angle* of your arms, the angle is 0 degrees. You are pointing in the exact same direction.
Cosine similarity looks at where the arms are pointing.

## The Mathematical Formula
$$ \text{Cosine Similarity} = \frac{A \cdot B}{||A|| \times ||B||} $$

Where:
- $A \cdot B$ is the Dot Product of the two vectors.
- $||A||$ is the magnitude (length) of vector A.

### The Scale
- **1.0**: The vectors point in the exact same direction (Identical meaning).
- **0.0**: The vectors are perpendicular at 90 degrees (Unrelated meaning).
- **-1.0**: The vectors point in exactly opposite directions (Opposite meaning).

## Week 2 assignment connection
To prove that our network learned relationships, we will extract the embedding for "Receive" and the embedding for "Restock", and calculate their Cosine Similarity. We expect it to be a high positive number (e.g. `0.85`), proving they clustered together in space.

## Small numerical example
Vector A: `[3, 4]` (Magnitude = 5)
Vector B: `[6, 8]` (Magnitude = 10)
- Dot Product: $(3 \times 6) + (4 \times 8) = 18 + 32 = 50$
- Product of Magnitudes: $5 \times 10 = 50$
- Cosine Similarity = $50 / 50 = 1.0$ (They point in the exact same direction!).

## Common misunderstanding
**Misunderstanding:** A Cosine Similarity of 0 means the words are opposites.
**Correction:** A similarity of 0 means they are completely unrelated (orthogonal). A similarity of -1 means they are opposites.

## Teach-back question
Why do we divide the dot product by the magnitudes of the vectors? What problem does that solve?

## My Understanding
*(Learner space to explain this in their own words later)*.

## Flashcards

Why is Cosine Similarity preferred over Euclidean distance (physical distance) for comparing word embeddings? #card
Because embeddings for common words can grow longer (higher magnitude) than rare words. Cosine similarity only measures the angle (direction) between vectors, ignoring their length, which is a better representation of semantic meaning.

What does a Cosine Similarity of 1.0, 0.0, and -1.0 mean respectively? #card
1.0 means identical direction (high similarity). 0.0 means orthogonal (unrelated). -1.0 means opposite direction (opposites).
