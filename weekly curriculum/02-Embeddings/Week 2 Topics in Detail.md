# Week 2 Topics in Detail

This document maps the core curriculum concepts to their permanent Deep-Dive notes.

## Stage 1: Data Preparation
[📖 Deep-Dive Note: 17 - LM - Tokenizers.md](../../topics/17%20-%20LM%20-%20Tokenizers.md)

**37. Tokenizer**: The separate, non-learning algorithm that breaks human text into strings and assigns them IDs before they enter the network.
**38. Whole-word vs Sub-word**: Whole-word maps complete words to IDs. Sub-word breaks them into pieces (like BPE) to handle unknown words gracefully.

---

## Stage 2: Learned Representations
[📖 Deep-Dive Note: 14 - LM - Embeddings.md](../../topics/14%20-%20LM%20-%20Embeddings.md)
[📖 Deep-Dive Note: 15 - NN - The Embedding Matrix.md](../../topics/15%20-%20NN%20-%20The%20Embedding%20Matrix.md)

**39. Embedding**: A dense vector of floats representing the "meaning" of a word, learned entirely through training context.
**40. The Embedding Matrix**: The 2D table holding all embedding vectors. Its shape is always `(Vocabulary Size, Embedding Dimension)`.
**41. Index Lookup**: The highly efficient $O(1)$ operation that replaces `One-Hot @ Matrix`. Mathematically identical, computationally superior.

---

## Stage 3: Measuring Similarity and Visualizing
[📖 Deep-Dive Note: 16 - MATH - Cosine Similarity.md](../../topics/16%20-%20MATH%20-%20Cosine%20Similarity.md)
[📖 Deep-Dive Note: 18 - MATH - Dimensionality Reduction and PCA.md](../../topics/18%20-%20MATH%20-%20Dimensionality%20Reduction%20and%20PCA.md)

**42. Cosine Similarity**: A metric from -1 to 1 that isolates the semantic direction (angle) between two vectors, ignoring their physical length.
**43. Dimensionality Reduction**: Compressing high-dimensional data (e.g. 8D) down to low dimensions (2D) for human visualization.
**44. PCA (Principal Component Analysis)**: A specific algorithm for finding the axes of greatest variance to project data onto a 2D plane.
