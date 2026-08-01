"""
week3_selfattention_numpy.py

Week 3 - Self-Attention from Scratch using NumPy

This example builds on week2_embeddings_numpy.py and demonstrates:
1. Embedding lookup for a short sequence
2. Query/Key/Value projections
3. Scaled dot-product self-attention
4. Context vector creation
5. Next-word prediction from attention context
6. Backpropagation through attention and output layers

Notes:
- We train only the
  output layer (W_out, b_out) while keeping embeddings and
  attention projection matrices fixed.
- This version trains embeddings plus Q/K/V and output layers.
- The model predicts the next word for the last token in each
	short input sequence.
"""

import numpy as np

np.random.seed(42)

# -------------------------------------------------------
# Vocabulary (same as Week 2)
# -------------------------------------------------------
vocab = [
	"Inventory",
	"Order",
	"Shipment",
	"Receive",
	"Restock",
	"Forecast",
	"Scenario",
]

word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 4
ATTN_DIM = 4
LEARNING_RATE = 0.08
EPOCHS = 4000
PRINT_EVERY = 400
GRAD_CLIP = 5.0

# -------------------------------------------------------
# Tiny sequence training corpus (context sequence -> next word)
# We use short sequences so self-attention has multiple tokens.
# -------------------------------------------------------
training_data = [
	(["Order", "Shipment"], "Receive"),
	(["Shipment", "Receive"], "Restock"),
	(["Receive", "Restock"], "Inventory"),
	(["Restock", "Inventory"], "Forecast"),
	(["Inventory", "Order"], "Shipment"),
	(["Order", "Receive"], "Inventory"),
	(["Shipment", "Restock"], "Inventory"),
]


# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------
def softmax(x, axis=-1):
	x_shifted = x - np.max(x, axis=axis, keepdims=True)
	e = np.exp(x_shifted)
	return e / np.sum(e, axis=axis, keepdims=True)


def cross_entropy(probabilities, target_idx):
	return -np.log(probabilities[target_idx] + 1e-12)


def cosine_similarity(a, b):
	return np.dot(a, b) / (
		np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
	)


# -------------------------------------------------------
# Model Parameters
# -------------------------------------------------------
# Token embeddings
embedding = np.random.randn(VOCAB_SIZE, EMBEDDING_DIM) * 0.1

# Self-attention projections
W_q = np.random.randn(EMBEDDING_DIM, ATTN_DIM) * 0.1
W_k = np.random.randn(EMBEDDING_DIM, ATTN_DIM) * 0.1
W_v = np.random.randn(EMBEDDING_DIM, ATTN_DIM) * 0.1

# Output layer (trained)
W_out = np.random.randn(ATTN_DIM, VOCAB_SIZE) * 0.1
b_out = np.zeros(VOCAB_SIZE)


# -------------------------------------------------------
# Forward Pass for One Sequence
# -------------------------------------------------------
def forward_sequence(input_words):
	"""
	input_words: list[str], e.g. ["Order", "Shipment"]

	Returns a cache with all intermediates and output probabilities.
	"""
	indices = [word_to_idx[w] for w in input_words]

	# X: [seq_len, embed_dim]
	X = embedding[indices]

	# Q, K, V: [seq_len, attn_dim]
	Q = X @ W_q
	K = X @ W_k
	V = X @ W_v

	# Attention scores: [seq_len, seq_len]
	scale = np.sqrt(ATTN_DIM)
	scores = (Q @ K.T) / scale

	# Row-wise softmax for attention weights
	attn_weights = softmax(scores, axis=1)

	# Context vectors for each token: [seq_len, attn_dim]
	context = attn_weights @ V

	# For next-word prediction, use the last token context
	h = context[-1]

	logits = h @ W_out + b_out
	probs = softmax(logits, axis=0)

	cache = {
		"indices": indices,
		"X": X,
		"Q": Q,
		"K": K,
		"V": V,
		"scores": scores,
		"attn_weights": attn_weights,
		"context": context,
		"h": h,
		"logits": logits,
		"probs": probs,
	}

	return cache


# -------------------------------------------------------
# Training Step
# -------------------------------------------------------
def train_step(input_words, target_word):
	"""
	Trains embedding, Q/K/V projections, and output projection.
	"""
	global embedding, W_q, W_k, W_v, W_out, b_out

	target_idx = word_to_idx[target_word]
	cache = forward_sequence(input_words)
	indices = cache["indices"]
	X = cache["X"]
	Q = cache["Q"]
	K = cache["K"]
	V = cache["V"]
	A = cache["attn_weights"]
	probs = cache["probs"]
	h = cache["h"]

	loss = cross_entropy(probs, target_idx)

	# dL/dlogits
	dlogits = probs.copy()
	dlogits[target_idx] -= 1.0

	# Output layer gradients
	dW_out = np.outer(h, dlogits)
	db_out = dlogits
	dh = W_out @ dlogits

	# Backprop into context (only last position contributes)
	seq_len = len(indices)
	dcontext = np.zeros((seq_len, ATTN_DIM))
	dcontext[-1] = dh

	# context = A @ V
	dA = dcontext @ V.T
	dV = A.T @ dcontext

	# Row-wise softmax backprop for attention matrix A = softmax(scores)
	row_dot = np.sum(dA * A, axis=1, keepdims=True)
	dscores = A * (dA - row_dot)

	# scores = (Q @ K.T) / sqrt(ATTN_DIM)
	scale = np.sqrt(ATTN_DIM)
	dQ = (dscores @ K) / scale
	dK = (dscores.T @ Q) / scale

	# Q = X @ W_q, K = X @ W_k, V = X @ W_v
	dW_q = X.T @ dQ
	dW_k = X.T @ dK
	dW_v = X.T @ dV

	dX = dQ @ W_q.T
	dX += dK @ W_k.T
	dX += dV @ W_v.T

	# Scatter token-position gradients back to embedding rows
	dEmbedding = np.zeros_like(embedding)
	for pos, idx in enumerate(indices):
		dEmbedding[idx] += dX[pos]

	# Gradient clipping to keep updates stable
	for grad in [dEmbedding, dW_q, dW_k, dW_v, dW_out, db_out]:
		np.clip(grad, -GRAD_CLIP, GRAD_CLIP, out=grad)

	# Gradient descent update
	embedding -= LEARNING_RATE * dEmbedding
	W_q -= LEARNING_RATE * dW_q
	W_k -= LEARNING_RATE * dW_k
	W_v -= LEARNING_RATE * dW_v
	W_out -= LEARNING_RATE * dW_out
	b_out -= LEARNING_RATE * db_out

	return loss, int(np.argmax(probs) == target_idx)


# -------------------------------------------------------
# Train
# -------------------------------------------------------
print("=" * 60)
print("Training Self-Attention (Week 3)...")
print("=" * 60)

for epoch in range(EPOCHS):
	# Shuffle order each epoch to reduce memorization-by-order.
	order = np.random.permutation(len(training_data))
	total_loss = 0.0
	total_correct = 0
	for i in order:
		inp_seq, target = training_data[i]
		loss, is_correct = train_step(inp_seq, target)
		total_loss += loss
		total_correct += is_correct

	if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
		acc = total_correct / len(training_data)
		print(f"Epoch {epoch:4d}   Loss = {total_loss:.4f}   Acc = {acc:.2%}")


# -------------------------------------------------------
# Predictions + Attention Visualization
# -------------------------------------------------------
print("\n" + "=" * 60)
print("Predictions with Attention Weights")
print("=" * 60)

correct = 0

for inp_seq, target in training_data:
	cache = forward_sequence(inp_seq)
	probs = cache["probs"]
	pred_idx = np.argmax(probs)
	pred_word = idx_to_word[pred_idx]
	correct += int(pred_word == target)

	print(
		f"Input: {inp_seq} -> Predicted: {pred_word:10s}  Expected: {target}"
	)

	# Print attention row for last token only
	last_row = cache["attn_weights"][-1]
	pairs = [
		f"{inp_seq[i]}:{last_row[i]:.3f}"
		for i in range(len(inp_seq))
	]
	print("  Last-token attention -> " + " | ".join(pairs))

final_acc = correct / len(training_data)
print(f"\nTraining set accuracy: {final_acc:.2%}")


# -------------------------------------------------------
# Embedding Similarity (carried over from Week 2 idea)
# -------------------------------------------------------
print("\n" + "=" * 60)
print("Embedding Cosine Similarity Matrix")
print("=" * 60)

header = " " * 12 + "".join(f"{w[:7]:>9}" for w in vocab)
print(header)

for w1 in vocab:
	row = f"{w1:12s}"
	for w2 in vocab:
		sim = cosine_similarity(
			embedding[word_to_idx[w1]],
			embedding[word_to_idx[w2]],
		)
		row += f"{sim:9.2f}"
	print(row)

print("\nDone.")
