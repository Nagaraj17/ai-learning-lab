"""
From Trained Next-Word Predictor to Embeddings

This script is a direct extension of Week 1 (01_Next_Word_Predictor.ipynb).

In Week 1, the model learned next-word transitions using one-hot input and dense layers.
In Week 2, we keep the same learning idea, but replace one-hot-only representation with learned embeddings.

By the end, we explain:
1. Why one-hot encoding does not capture similarity
2. What an embedding is
3. Where embeddings are stored
4. How embeddings are learned
5. Why similar words end up with similar vectors
"""

# Step 0 -- Imports and reproducibility
import numpy as np

np.random.seed(42)

# Step 1 -- Vocabulary and index mapping (same as Week 1)
VOCAB = [
    "Inventory",
    "Order",
    "Cabinet",
    "Drug",
    "Invoice",
    "Shipment",
    "Receive",
    "Restock",
    "Forecast",
    "Scenario",
]

# Same index mapping used in Week 1
word_to_idx = {
    "Inventory": 0,
    "Order": 1,
    "Cabinet": 2,
    "Drug": 3,
    "Invoice": 4,
    "Shipment": 5,
    "Receive": 6,
    "Restock": 7,
    "Forecast": 8,
    "Scenario": 9,
}

idx_to_word = {i: w for w, i in word_to_idx.items()}
VOCAB_SIZE = len(VOCAB)

print("Week 1 index mapping reused:")
for word in sorted(VOCAB, key=lambda w: word_to_idx[w]):
    print(f"{word:10s} -> {word_to_idx[word]}")

# Step 2 -- Training data and helper functions
training_data = [
    ("Order", "Shipment"),
    ("Shipment", "Receive"),
    ("Receive", "Restock"),
    ("Restock", "Inventory"),
    ("Inventory", "Forecast"),
    ("Forecast", "Order"),
    ("Invoice", "Scenario"),
    ("Cabinet", "Drug"),
    ("Drug", "Invoice"),
]

EMBEDDING_DIM = 6
LEARNING_RATE = 0.1
EPOCHS = 1500


def one_hot(index):
    v = np.zeros(VOCAB_SIZE)
    v[index] = 1.0
    return v


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def cross_entropy(probabilities, target_idx):
    return -np.log(probabilities[target_idx] + 1e-12)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


# Step 3 -- Baseline check: why one-hot does not capture similarity
print("\nStep 3: One-hot similarity check")
pairs = [("Receive", "Restock"), ("Receive", "Scenario")]
for a, b in pairs:
    va = one_hot(word_to_idx[a])
    vb = one_hot(word_to_idx[b])
    print(a, b, "dot=", float(np.dot(va, vb)), "euclidean=", float(np.linalg.norm(va - vb)))


# Step 4 -- Embedding layer and model parameters
embedding = np.random.randn(VOCAB_SIZE, EMBEDDING_DIM) * 0.1
W = np.random.randn(EMBEDDING_DIM, VOCAB_SIZE) * 0.1
b = np.zeros(VOCAB_SIZE)

print("\nStep 4: Parameter shapes")
print("embedding.shape =", embedding.shape)
print("W.shape =", W.shape)


# Step 5 -- Forward pass and gradient update (how embeddings learn)
def forward(input_idx):
    x = one_hot(input_idx)
    e = x @ embedding
    logits = e @ W + b
    probs = softmax(logits)
    return x, e, logits, probs


def train_step(input_word, target_word):
    global embedding, W, b

    input_idx = word_to_idx[input_word]
    target_idx = word_to_idx[target_word]

    x, e, _, probs = forward(input_idx)
    loss = cross_entropy(probs, target_idx)

    dlogits = probs.copy()
    dlogits[target_idx] -= 1.0

    dW = np.outer(e, dlogits)
    db = dlogits
    de = W @ dlogits
    d_embedding = np.outer(x, de)

    W -= LEARNING_RATE * dW
    b -= LEARNING_RATE * db
    embedding -= LEARNING_RATE * d_embedding

    return loss


# Step 6 -- Training loop (repeat over many epochs)
print("\nStep 6: Training")
for epoch in range(EPOCHS):
    total_loss = 0.0
    for inp, target in training_data:
        total_loss += train_step(inp, target)
    if epoch % 150 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:4d}  Loss={total_loss:.4f}")


# Step 7 -- Inspect learned embeddings
print("\nStep 7: Learned embeddings")
for word in sorted(VOCAB, key=lambda w: word_to_idx[w]):
    vec = np.round(embedding[word_to_idx[word]], 3)
    print(f"idx={word_to_idx[word]:2d}  {word:10s} : {vec}")


# Step 8 -- Nearest neighbors with cosine similarity
def top_k_neighbors(word, k=3):
    vec = embedding[word_to_idx[word]]
    scores = []
    for other in VOCAB:
        if other == word:
            continue
        sim = cosine_similarity(vec, embedding[word_to_idx[other]])
        scores.append((other, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


print("\nStep 8: Nearest neighbors")
for word in sorted(VOCAB, key=lambda w: word_to_idx[w]):
    print(word, "->", top_k_neighbors(word, 2))


# Step 9 -- Focused comparison for your goal
sim_receive_restock = cosine_similarity(
    embedding[word_to_idx["Receive"]], embedding[word_to_idx["Restock"]]
)
sim_receive_scenario = cosine_similarity(
    embedding[word_to_idx["Receive"]], embedding[word_to_idx["Scenario"]]
)

print("\nStep 9: Focused cosine comparison")
print("Receive~Restock cosine:", round(float(sim_receive_restock), 3))
print("Receive~Scenario cosine:", round(float(sim_receive_scenario), 3))


# Final takeaway
print("\nFinal takeaway (Week 2 objectives answered)")
print("1. One-hot vectors only identify words; they do not encode closeness between meanings.")
print("2. An embedding is a trainable dense vector representation for each word.")
print("3. Embeddings are stored as rows in the embedding matrix.")
print("4. During backpropagation, gradients update those rows to reduce prediction loss.")
print("5. Words used in similar contexts get similar updates, so their vectors become close under cosine similarity.")
print("This is the bridge from Week 1 prediction learning to Week 2 representation learning.")
