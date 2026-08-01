"""
Week 3 - Attention (standalone script)
Focus is Week 3. The short Week 2 block at the top only exists so we have
trained embeddings for attention to work on.
Scope: one attention step, one head, forward only.
"""

import numpy as np
np.set_printoptions(precision=3, suppress=True)
np.random.seed(42)


# =====================================================================
# PREREQUISITE FROM WEEK 2 (condensed) - just enough to get trained embeddings
# =====================================================================
vocab = ["Inventory", "Order", "Cabinet", "Drug", "Invoice",
         "Shipment", "Receive", "Restock", "Forecast", "Scenario"]
word_to_id = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
embed_dim, hidden_dim = 4, 16

# the same warehouse cycle as Weeks 1-2
pairs = [("Order","Shipment"), ("Shipment","Receive"), ("Receive","Restock"),
         ("Restock","Inventory"), ("Inventory","Forecast"), ("Forecast","Order")]
X_ids = np.array([word_to_id[a] for a, _ in pairs])
Y_ids = np.array([word_to_id[b] for _, b in pairs])

Embedding = np.random.randn(vocab_size, embed_dim) * 0.5
W1 = np.random.randn(embed_dim, hidden_dim) * 0.5
b1 = np.zeros(hidden_dim)
W2 = np.random.randn(hidden_dim, vocab_size) * 0.5
b2 = np.zeros(vocab_size)

def softmax_rows(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# quick train so the embeddings are meaningful, not random (same loop as Week 2)
for _ in range(3000):
    emb = Embedding[X_ids]
    a1 = np.tanh(emb @ W1 + b1)
    probs = softmax_rows(a1 @ W2 + b2)
    n = len(Y_ids)
    dz2 = probs.copy(); dz2[np.arange(n), Y_ids] -= 1; dz2 /= n
    dW2 = a1.T @ dz2; db2 = dz2.sum(0)
    dz1 = (dz2 @ W2.T) * (1 - a1**2)
    dW1 = emb.T @ dz1; db1 = dz1.sum(0)
    demb = dz1 @ W1.T
    dEmb = np.zeros_like(Embedding); np.add.at(dEmb, X_ids, demb)
    Embedding -= 0.5*dEmb; W1 -= 0.5*dW1; b1 -= 0.5*db1; W2 -= 0.5*dW2; b2 -= 0.5*db2

print("Week 2 prerequisite done: embeddings are trained.\n")


# =====================================================================
# WEEK 3 STARTS HERE
# =====================================================================
def softmax_1d(z):
    e = np.exp(z - z.max())
    return e / e.sum()

# ---- Step 1: feed a SEQUENCE instead of one word ----
# Attention needs several words to weigh, so we stack their embeddings.
sequence = ["Receive", "Order", "Shipment"]
X = Embedding[[word_to_id[w] for w in sequence]]   # (seq_len, embed_dim)
print("Sequence:", sequence)
print("X shape (words x embed_dim):", X.shape, "\n")

# ---- Step 2: make Query / Key / Value ----
# Each word plays 3 roles. Wq/Wk/Wv produce those 3 versions.
# They are random here (untrained) - exactly like embeddings were in Week 2.
np.random.seed(0)
Wq = np.random.randn(embed_dim, embed_dim) * 0.5
Wk = np.random.randn(embed_dim, embed_dim) * 0.5
Wv = np.random.randn(embed_dim, embed_dim) * 0.5
Q, K, V = X @ Wq, X @ Wk, X @ Wv

# ---- Step 3: ONE attention step (the last word does the predicting) ----
# score = query . key (similarity) -> scale -> softmax -> blend the values
scores  = (Q[-1] @ K.T) / np.sqrt(embed_dim)
weights = softmax_1d(scores)
context = weights @ V

print("Attention weights (last word listening to each word):")
for w, word in zip(weights, sequence):
    print(f"   {word:9s}: {w*100:5.1f}%")
print("Context vector (blended summary):", context, "\n")

# ---- Step 4: plug the context into the Week 2 predictor ----
# Only change vs Week 2: the hidden layer eats 'context' instead of one embedding.
a1 = np.tanh(context @ W1 + b1)
probs = softmax_1d(a1 @ W2 + b2)
print("Next-word probabilities:")
for i in np.argsort(probs)[::-1]:
    print(f"   {vocab[i]:10s}: {probs[i]*100:5.1f}%")
print()

# ---- Step 5: self-attention as a text grid (every word queries every word) ----
S = (Q @ K.T) / np.sqrt(embed_dim)
Wmat = np.stack([softmax_1d(r) for r in S])
print("Self-attention grid (row = who asks, col = who they listen to):")
print("           " + "".join(f"{w:>9s}" for w in sequence))
for i, word in enumerate(sequence):
    print(f"{word:>10s} " + "".join(f"{Wmat[i,j]*100:8.1f}%" for j in range(len(sequence))))

# NOTE: Wq/Wk/Wv are untrained, so the weights are focused but arbitrary.
# The mechanism is what matters this week; training them is a later step.
