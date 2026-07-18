# Dimensionality Reduction and PCA

## One-line definition
**Dimensionality Reduction** is a mathematical technique to compress high-dimensional vectors into fewer dimensions (like 2D or 3D) while preserving as much of the original semantic distance between the vectors as possible.

## Why it exists
In modern neural networks, embedding vectors can have 1,000 or more dimensions (e.g., $W_{embed}$ shape `(50000, 1024)`). Humans cannot visualize a 1,024-dimensional space. We need a way to squash these vectors down to a 2D graph (X and Y coordinates) so we can see which words clustered together.

## Beginner intuition
Imagine you have a 3D physical model of a city, and you want to put it in a 2D book. You take a photograph.
If you take a photo from directly above (a satellite view), you preserve the distances between the streets perfectly, but you lose the heights of the buildings. If you take a photo from the side, you see the heights, but lose the street map.
Dimensionality Reduction is the math of finding the "best camera angle" to take a 2D photo of a massive, multi-dimensional object so that the most important relationships are visible.

## PCA (Principal Component Analysis)
PCA is the most classic algorithm for this. It mathematically analyzes the data to find the "axes of greatest variance" (the camera angle where the data is most spread out).
If you have 8-dimensional embeddings, PCA will calculate the two most important mathematical directions that separate the words, and project the data onto those two directions, giving you an X and Y coordinate.

## Week 2 assignment connection
In our stretch goal, we learn embeddings with an 8-dimensional hidden size. To visualize if "Receive" and "Restock" actually clustered together, we will use the `scikit-learn` library to run PCA, transforming our `(N, 8)` matrix into an `(N, 2)` matrix. We can then plot these 2D points on a standard scatter plot.

## Common misunderstanding
**Misunderstanding:** The 2D PCA graph is a perfect representation of the embeddings.
**Correction:** Dimensionality reduction always loses information (just like a 2D photo loses the 3D depth). Two words might look close together on a 2D PCA plot, but actually be far apart in the true 1,024-dimensional space. Always trust the Cosine Similarity calculation over the 2D visual plot.

## Teach-back question
If a 2D PCA plot shows "Dog" and "Computer" right next to each other, but their Cosine Similarity in the original 8-dimensional space is 0.0, which one is telling the truth?

## My Understanding
*(Learner space to explain this in their own words later)*.

## Flashcards

What is the primary purpose of Dimensionality Reduction (like PCA) in machine learning? #card
To compress high-dimensional data (like 1,024-dimension embeddings) down to 2 or 3 dimensions so that humans can visually plot and interpret the relationships.

Does a 2D PCA plot perfectly preserve the distances between embeddings? #card
No. Dimensionality reduction always loses information. Vectors that appear close on a 2D plot might actually be separated in a higher dimension that was flattened.
