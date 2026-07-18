# Review: Week 2 (Embeddings)

Before proceeding to Week 3, ensure you can confidently answer the following questions without checking the notes. 
*(If you are unsure of any answer, consult the `topics/` directory).*

---

## 1. Conceptual Knowledge
1. **Why does One-Hot Encoding fail to capture semantic meaning?**
2. **Where does a Neural Network get the values for an Embedding Vector?**
3. **What is the difference between Whole-Word tokenization and Sub-Word tokenization?**
4. **Why is Cosine Similarity preferred over Euclidean distance when comparing embeddings?**
5. **Why can't we trust a 2D PCA graph completely?**

---

## 2. Code Comprehension
Look at the following lines of code from the Week 2 assignment. What exactly do they do?

1. `vocab = {"Receive": 0, "Restock": 1, ...}`
2. `W_embed = np.random.randn(vocab_size, embedding_dim)`
3. `embedding = W_embed[x_token_id]`
4. `dot_product = np.dot(emb1, emb2)`
5. `cosine_sim = dot_product / (np.linalg.norm(emb1) * np.linalg.norm(emb2))`
6. `pca = PCA(n_components=2); coords = pca.fit_transform(W_embed)`

---

## 3. Mathematical Calculations
Without using code, calculate the following:

1. **Cosine Similarity:** If Vector A is `[1, 0]` and Vector B is `[0, 1]`, what is their Cosine Similarity? What does that mean?
2. **Matrix Shape:** If your vocabulary size is $10,000$ and you want an embedding dimension of $256$, what is the exact shape of your Embedding Matrix?
3. **Lookup Math:** If `x = [0, 1, 0]` and `W` is a $3 \times 4$ matrix, write out the array indexing equivalent of `x @ W`.

---

## 4. Debugging & What-If Scenarios
What would happen to the neural network if you made these changes?

1. You pass the word `"Apple"` into the network, but `"Apple"` is not in your hardcoded `vocab` dictionary.
2. You initialize your Embedding Matrix `W_embed` with `np.zeros()` instead of `np.random.randn()`.
3. You calculate Cosine Similarity between two identical vectors `[2, 3]` and `[2, 3]`. What will the result be?
