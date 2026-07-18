# PREREQUISITE KNOWLEDGE (Week 2)

This is the consolidated study guide for Week 2 (Embeddings and Semantic Relationships). It covers every new concept required for the assignment.

> [!NOTE]
> Ensure you have read and understood the required Week 1 concepts before starting this week:
> - [02 - LM - Tokens and Vocabulary](../../topics/02%20-%20LM%20-%20Tokens%20and%20Vocabulary.md)
> - [03 - LM - One-Hot Encoding](../../topics/03%20-%20LM%20-%20One-Hot%20Encoding.md)
> - [10 - NN - Gradients and Backpropagation](../../topics/10%20-%20NN%20-%20Gradients%20and%20Backpropagation.md)

---

## 1. Tokenizers
Before text can be embedded, it must be turned into numbers. A Tokenizer handles this translation based on a strict set of rules. In our toy problem, we use **Whole-Word Tokenization**, mapping exact strings to exact integers (e.g., `"Receive" -> 6`). Modern networks use **Sub-Word Tokenization** to handle unseen words, breaking them into smaller chunks. 
**Crucial Concept:** The tokenizer is a fixed algorithm. It does not "learn" during backpropagation.

## 2. Embeddings
An Embedding is a dense vector (a list of numbers) that mathematically represents the meaning of a token. Unlike a One-Hot vector (`[0, 1, 0, 0]`), which is mostly empty space, an embedding (`[0.72, -0.15, 0.44]`) uses every dimension to capture a different "attribute" of the word. The network learns these attributes entirely on its own during training.

## 3. The Embedding Matrix
All the embedding vectors for our vocabulary are stored in a single matrix called the Embedding Matrix. If our vocabulary is 10 words, and we want 8 attributes per word, our matrix shape is `(10, 8)`. 
In Week 1, we multiplied a one-hot vector by a weight matrix to select a row. In Week 2, we just use the Token ID as an array index (`matrix[token_id]`) to pull out the row instantly, bypassing the massive waste of multiplying by zeros.

## 4. Cosine Similarity
To measure if two words mean similar things, we extract their embedding vectors and measure the *angle* between them using Cosine Similarity.
- **1.0**: Pointing in the exact same direction (Identical meaning).
- **0.0**: Orthogonal/Perpendicular (Unrelated meaning).
- **-1.0**: Pointing opposite directions (Opposite meaning).
We divide the dot product by the magnitudes to ensure we are only comparing the angle, ignoring the length of the vectors.

## 5. Dimensionality Reduction (PCA)
Embeddings are high-dimensional (e.g., 8 dimensions). Humans can only see in 3D. We use a mathematical technique like PCA (Principal Component Analysis) to squash the 8 dimensions down to 2 dimensions so we can plot them on a scatter graph (X and Y coordinates). This lets us visually prove that "Receive" and "Restock" clustered together!

---
> [!TIP]
> 🧮 **Manual Exercises**: Do not proceed to the Jupyter Notebook until you have manually verified the math:
> - [Level 2: Embedding Lookup](manual-exercises/01_EMBEDDING_LOOKUP.md)
> - [Level 2: Cosine Similarity](manual-exercises/02_COSINE_SIMILARITY.md)
