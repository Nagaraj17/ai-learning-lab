import numpy as np

# =============================================================================
# Week 2 - Embedding Exercise  (NumPy only)
# =============================================================================
# 1) Learning explicit embeddings (embedding matrix E)
# 2) Training next-word prediction with gradient descent
# 3) Printing embedding table and similarity views

np.set_printoptions(precision=4, suppress=True)


# ==============================================
# Step 0 -- Vocabulary and training transitions 
# ==============================================

vocabulary = [
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

token_to_id = {token: i for i, token in enumerate(vocabulary)}
id_to_token = {i: token for token, i in token_to_id.items()}
vocabulary_size = len(vocabulary)

# Training transitions specified in the exercise.
true_next = {
    "Order": "Shipment",
    "Shipment": "Receive",
    "Receive": "Restock",
    "Restock": "Inventory",
    "Inventory": "Forecast",
    "Forecast": "Order",
}

print("=" * 80)
print("EMBEDDING EXERCISE - DATA SETUP")
print("=" * 80)
print("Vocabulary:", vocabulary)
print("Training transitions:")
for src, dst in true_next.items():
    print(f"  {src:10s} -> {dst}")
print()


# =============================================================================
# Step 1 -- Model definition: explicit embedding matrix + output layer
# =============================================================================
# E[token_id] gives embedding vector for that token.

np.random.seed(42)
embedding_dim = 6

E = np.random.randn(vocabulary_size, embedding_dim) * 0.2
W_out = np.random.randn(embedding_dim, vocabulary_size) * 0.2
b_out = np.zeros(vocabulary_size)

print("Model architecture:")
print(f"  Embedding matrix E shape: {E.shape}  (vocab_size x embedding_dim)")
print(f"  Output weight W_out shape: {W_out.shape}  (embedding_dim x vocab_size)")
print(f"  Output bias b_out shape: {b_out.shape}")
print()


# =============================================================================
# Step 2 -- Helper functions
# =============================================================================

def softmax(logits):
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def forward(token):
    token_id = token_to_id[token]
    embedding = E[token_id]                    # explicit embedding lookup
    logits = embedding @ W_out + b_out
    probs = softmax(logits)
    return token_id, embedding, logits, probs


def classify_similarity(cos_val):
    if cos_val >= 0.70:
        return "Strongly aligned"
    if cos_val >= 0.20:
        return "Moderately aligned"
    if cos_val <= -0.20:
        return "Opposite direction"
    return "Weak/neutral"


# =============================================================================
# Step 3 -- Baseline prediction before training
# =============================================================================

print("=" * 80)
print("BASELINE (Before Training)")
print("=" * 80)

correct = 0
for src in true_next:
    _, _, _, probs = forward(src)
    pred = id_to_token[int(np.argmax(probs))]
    truth = true_next[src]
    confidence = float(np.max(probs))
    mark = "OK" if pred == truth else "XX"
    correct += int(pred == truth)
    print(f"{mark} {src:10s} -> guess: {pred:10s} ({confidence:5.1%})  truth: {truth}")
print(f"Baseline accuracy: {correct}/{len(true_next)}")
print()


# =============================================================================
# Step 4 -- Prepare training pairs
# =============================================================================

X_ids = [token_to_id[src] for src in true_next.keys()]
Y_ids = [token_to_id[dst] for dst in true_next.values()]


# =============================================================================
# Step 5 -- Train with backpropagation + gradient descent
# =============================================================================

learning_rate = 0.15
epochs = 2500

print("=" * 80)
print("TRAINING")
print("=" * 80)

for epoch in range(epochs):
    total_loss = 0.0

    for x_id, y_id in zip(X_ids, Y_ids):
        # Forward
        embedding = E[x_id]
        logits = embedding @ W_out + b_out
        probs = softmax(logits)

        # Cross-entropy loss
        loss = -np.log(probs[y_id] + 1e-9)
        total_loss += loss

        # Backward
        dlogits = probs.copy()
        dlogits[y_id] -= 1.0

        dW_out = np.outer(embedding, dlogits)
        db_out = dlogits

        # Error signal back to embedding vector for current input token
        dembedding = W_out @ dlogits

        # Gradient descent updates
        W_out -= learning_rate * dW_out
        b_out -= learning_rate * db_out
        E[x_id] -= learning_rate * dembedding

    if epoch % 500 == 0 or epoch == epochs - 1:
        avg_loss = total_loss / len(X_ids)
        print(f"Epoch {epoch:4d} | Loss: {total_loss:.6f} | Avg Loss: {avg_loss:.6f}")

print()


# =============================================================================
# Step 6 -- Accuracy after training
# =============================================================================

print("=" * 80)
print("AFTER TRAINING")
print("=" * 80)

correct = 0
for src in true_next:
    _, _, _, probs = forward(src)
    pred = id_to_token[int(np.argmax(probs))]
    truth = true_next[src]
    confidence = float(np.max(probs))
    mark = "OK" if pred == truth else "XX"
    correct += int(pred == truth)
    print(f"{mark} {src:10s} -> guess: {pred:10s} ({confidence:5.1%})  truth: {truth}")

print(f"Final accuracy: {correct}/{len(true_next)}")
print()


# =============================================================================
# Step 7 -- Learned embedding table
# =============================================================================

print("=" * 80)
print("LEARNED EMBEDDING TABLE")
print("=" * 80)

for token in vocabulary:
    vec = E[token_to_id[token]]
    formatted = " ".join(f"{v: .4f}" for v in vec)
    seen_flag = "(trained)" if token in true_next or token in true_next.values() else "(unseen in transitions)"
    print(f"{token:10s} {seen_flag:24s} [{formatted}]")
print()


# =============================================================================
# Step 8 -- Cosine similarity matrix
# =============================================================================

print("=" * 80)
print("COSINE SIMILARITY MATRIX")
print("=" * 80)

header = " " * 12 + " ".join(f"{name[:8]:>9s}" for name in vocabulary)
print(header)
print("-" * len(header))

for token_a in vocabulary:
    vec_a = E[token_to_id[token_a]]
    row = []
    for token_b in vocabulary:
        vec_b = E[token_to_id[token_b]]
        row.append(f"{cosine_similarity(vec_a, vec_b):9.4f}")
    print(f"{token_a:10s} " + " ".join(row))
print()

print("=" * 80)

receive_vec = E[token_to_id["Receive"]]
restock_vec = E[token_to_id["Restock"]]
scenario_vec = E[token_to_id["Scenario"]]

sim_receive_restock = cosine_similarity(receive_vec, restock_vec)
sim_receive_scenario = cosine_similarity(receive_vec, scenario_vec)

print(f"Similarity(Receive, Restock): {sim_receive_restock:.4f}")
print(f"Similarity(Receive, Scenario): {sim_receive_scenario:.4f}")

if sim_receive_restock > sim_receive_scenario:
    print("Result: Receive is closer to Restock than to Scenario (objective trend achieved).")
else:
    print("Result: In this run, objective trend is weak/inverted; add more contextual data or epochs.")
print()


# =============================================================================
# Step 9 -- Nearest neighbors and 2D coordinates
# =============================================================================

print("=" * 80)
print("TOP-3 NEAREST TOKENS (by cosine similarity)")
print("=" * 80)

for token in vocabulary:
    idx = token_to_id[token]
    vec = E[idx]

    sims = []
    for other in vocabulary:
        if other == token:
            continue
        other_vec = E[token_to_id[other]]
        sims.append((other, cosine_similarity(vec, other_vec)))

    sims.sort(key=lambda item: item[1], reverse=True)
    top3 = sims[:3]
    desc = ", ".join(f"{name} ({score:.3f}, {classify_similarity(score)})" for name, score in top3)
    print(f"{token:10s} -> {desc}")
print()

# Simple PCA-style 2D projection for visual table.
centered = E - np.mean(E, axis=0, keepdims=True)
cov = np.cov(centered, rowvar=False)
e_vals, e_vecs = np.linalg.eigh(cov)
order = np.argsort(e_vals)[::-1]
proj2 = centered @ e_vecs[:, order[:2]]

print("=" * 80)
print("2D COORDINATES (PCA-style projection)")
print("=" * 80)
print("Token       |      PC1 |      PC2")
print("-" * 34)
for token, p in zip(vocabulary, proj2):
    print(f"{token:10s} | {p[0]:8.4f} | {p[1]:8.4f}")
print()

