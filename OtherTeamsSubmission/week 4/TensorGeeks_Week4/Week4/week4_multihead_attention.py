"""
week4_multihead_attention.py

Week 4 - Multi-Head Self-Attention from Scratch using NumPy

What this script demonstrates:
1. Multiple independent attention heads (4 heads)
2. Each head computes its own Q/K/V projections
3. Head outputs are concatenated and linearly projected
4. Residual connection from the last token embedding
5. Per-head attention visualization and comparison

Notes:
- This is a teaching script, so the network is intentionally small.
- The prediction of next token is based on the 4-token input sequence.
"""

import numpy as np

np.random.seed(42)


# -------------------------------------------------------
# Vocabulary (medical claims domain)
# -------------------------------------------------------
tokens = [
	"PracticeA",
	"PracticeB",
	"DrugA",
	"DrugB",
	"PayerA",
	"PayerB",
	"Monday",
	"Friday",
	"Approved",
	"Denied",
]

word_to_idx = {w: i for i, w in enumerate(tokens)}
idx_to_word = {i: w for w, i in word_to_idx.items()}

VOCAB_SIZE = len(tokens)


# -------------------------------------------------------
# Hyperparameters
# -------------------------------------------------------
D_MODEL = 8
NUM_HEADS = 4
HEAD_DIM = 4
LEARNING_RATE = 0.05
EPOCHS = 6000
PRINT_EVERY = 600
GRAD_CLIP = 5.0
NUMERIC_EPS = 1e-12


# -------------------------------------------------------
# Training data: medical claims decision patterns
# Input length is 4 tokens (Practice, Drug, Payer, Day), target is decision.
# -------------------------------------------------------
training_data = [
	(["PracticeA", "DrugA", "PayerA", "Monday"], "Approved"),
	(["PracticeA", "DrugB", "PayerB", "Monday"], "Denied"),
	(["PracticeB", "DrugA", "PayerA", "Friday"], "Denied"),
	(["PracticeB", "DrugB", "PayerA", "Monday"], "Approved"),
]


# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------
def softmax(x, axis=-1):
	x_shifted = x - np.max(x, axis=axis, keepdims=True)
	e = np.exp(x_shifted)
	return e / np.sum(e, axis=axis, keepdims=True)


def cross_entropy(probabilities, target_idx):
	return -np.log(probabilities[target_idx] + NUMERIC_EPS)


# Layer normalization removed for simplicity
# (Residual connection provides sufficient stability)


def cosine_similarity(a, b):
	return np.dot(a, b) / (
		np.linalg.norm(a) * np.linalg.norm(b) + NUMERIC_EPS
	)


def entropy(p):
	return -np.sum(p * np.log(p + NUMERIC_EPS))


def ascii_heat(values, width=24):
	"""
	Convert values in [0,1] to compact ASCII bars.
	"""
	bars = []
	for v in values:
		filled = int(round(v * width))
		bars.append("#" * filled + "." * (width - filled))
	return bars


# -------------------------------------------------------
# Model parameters
# -------------------------------------------------------
embedding = np.random.randn(VOCAB_SIZE, D_MODEL) * 0.1

# One set of Q/K/V projection matrices per head
W_q = np.random.randn(NUM_HEADS, D_MODEL, HEAD_DIM) * 0.1
W_k = np.random.randn(NUM_HEADS, D_MODEL, HEAD_DIM) * 0.1
W_v = np.random.randn(NUM_HEADS, D_MODEL, HEAD_DIM) * 0.1

# Projection after concatenating all head outputs
W_o = np.random.randn(NUM_HEADS * HEAD_DIM, D_MODEL) * 0.1

# Final classifier
W_out = np.random.randn(D_MODEL, VOCAB_SIZE) * 0.1
b_out = np.zeros(VOCAB_SIZE)


# -------------------------------------------------------
# Forward pass
# -------------------------------------------------------
def forward_sequence(input_words):
	indices = [word_to_idx[w] for w in input_words]
	token_embeddings = embedding[indices]  # [seq_len, D_MODEL]

	head_caches = []
	head_outputs = []
	attn_scale = np.sqrt(HEAD_DIM)

	for h in range(NUM_HEADS):
		Q = token_embeddings @ W_q[h]  # [seq_len, HEAD_DIM]
		K = token_embeddings @ W_k[h]
		V = token_embeddings @ W_v[h]

		scores = (Q @ K.T) / attn_scale
		attention_weights = softmax(scores, axis=1)  # [seq_len, seq_len]
		context = attention_weights @ V  # [seq_len, HEAD_DIM]

		head_output = context[-1]  # last-token context per head
		head_outputs.append(head_output)
		head_caches.append(
			{
				"Q": Q,
				"K": K,
				"V": V,
				"scores": scores,
				"A": attention_weights,
				"context": context,
			}
		)

	concat = np.concatenate(head_outputs, axis=0)  # [NUM_HEADS * HEAD_DIM]
	z = concat @ W_o  # [D_MODEL]

	# Residual connection (layer norm removed for simplicity)
	h = token_embeddings[-1] + z

	logits = h @ W_out + b_out
	probs = softmax(logits, axis=0)

	cache = {
		"indices": indices,
		"X": token_embeddings,
		"head_caches": head_caches,
		"head_outputs": head_outputs,
		"concat": concat,
		"z": z,

		"h": h,
		"logits": logits,
		"probs": probs,
	}
	return cache


# -------------------------------------------------------
# Backward + update
# -------------------------------------------------------
def train_step(input_words, target_word):
	global embedding, W_q, W_k, W_v, W_o, W_out, b_out

	target_idx = word_to_idx[target_word]
	cache = forward_sequence(input_words)

	indices = cache["indices"]
	X = cache["X"]
	head_caches = cache["head_caches"]
	probs = cache["probs"]
	concat = cache["concat"]
	h = cache["h"]

	loss = cross_entropy(probs, target_idx)

	# dL/dlogits
	dlogits = probs.copy()
	dlogits[target_idx] -= 1.0

	# Output layer gradients
	dW_out = np.outer(h, dlogits)
	db_out = dlogits
	dh = W_out @ dlogits

	# Backprop through residual connection
	# h = X[-1] + z
	dz = dh
	dX = np.zeros_like(X)
	dX[-1] += dh

	# z = concat @ W_o
	dW_o = np.outer(concat, dz)
	dconcat = dz @ W_o.T

	# Split concatenated head gradients
	dhead_outputs = dconcat.reshape(NUM_HEADS, HEAD_DIM)

	dW_q = np.zeros_like(W_q)
	dW_k = np.zeros_like(W_k)
	dW_v = np.zeros_like(W_v)

	attn_scale = np.sqrt(HEAD_DIM)

	for h_idx in range(NUM_HEADS):
		head = head_caches[h_idx]
		Q = head["Q"]
		K = head["K"]
		V = head["V"]
		A = head["A"]
		context = head["context"]

		# Only last context row contributes to classification
		dcontext = np.zeros_like(context)
		dcontext[-1] = dhead_outputs[h_idx]

		# context = A @ V
		dA = dcontext @ V.T
		dV_local = A.T @ dcontext

		# Row-wise softmax backward
		row_dot = np.sum(dA * A, axis=1, keepdims=True)
		dscores = A * (dA - row_dot)

		dQ = (dscores @ K) / attn_scale
		dK = (dscores.T @ Q) / attn_scale

		dW_q[h_idx] = X.T @ dQ
		dW_k[h_idx] = X.T @ dK
		dW_v[h_idx] = X.T @ dV_local

		dX += dQ @ W_q[h_idx].T
		dX += dK @ W_k[h_idx].T
		dX += dV_local @ W_v[h_idx].T

	# Scatter token-position gradients to embedding rows
	dEmbedding = np.zeros_like(embedding)
	for pos, idx in enumerate(indices):
		dEmbedding[idx] += dX[pos]

	# Gradient clipping for stability
	for grad in [dEmbedding, dW_q, dW_k, dW_v, dW_o, dW_out, db_out]:
		np.clip(grad, -GRAD_CLIP, GRAD_CLIP, out=grad)

	# Parameter update
	embedding -= LEARNING_RATE * dEmbedding
	W_q -= LEARNING_RATE * dW_q
	W_k -= LEARNING_RATE * dW_k
	W_v -= LEARNING_RATE * dW_v
	W_o -= LEARNING_RATE * dW_o
	W_out -= LEARNING_RATE * dW_out
	b_out -= LEARNING_RATE * db_out

	return loss, int(np.argmax(probs) == target_idx)


# -------------------------------------------------------
# Training / Evaluation Helpers
# -------------------------------------------------------
def run_training():
	print("=" * 70)
	print("Week 4: Multi-Head Self-Attention Training")
	print("=" * 70)
	print(f"Heads: {NUM_HEADS} | D_MODEL: {D_MODEL} | HEAD_DIM: {HEAD_DIM}")

	for epoch in range(EPOCHS):
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


def evaluate_and_visualize():
	print("\n" + "=" * 70)
	print("Predictions and Per-Head Attention (Last Token Query)")
	print("=" * 70)

	correct = 0
	head_signature_vectors = [[] for _ in range(NUM_HEADS)]
	head_entropies = [[] for _ in range(NUM_HEADS)]

	for inp_seq, target in training_data:
		cache = forward_sequence(inp_seq)
		probs = cache["probs"]
		pred_idx = int(np.argmax(probs))
		pred_word = idx_to_word[pred_idx]
		correct += int(pred_word == target)

		print(
			f"Input: {inp_seq} -> Predicted: {pred_word:10s}  Expected: {target}"
		)

		for h_idx, head in enumerate(cache["head_caches"]):
			last_row = head["A"][-1]  # attention from last token to all tokens
			head_signature_vectors[h_idx].append(last_row)
			head_entropies[h_idx].append(entropy(last_row))

			bars = ascii_heat(last_row, width=20)
			parts = []
			for token, weight, bar in zip(inp_seq, last_row, bars):
				parts.append(f"{token:10s} {weight:.3f} |{bar}|")

			print(f"  Head {h_idx + 1}:")
			for p in parts:
				print(f"    {p}")

	final_acc = correct / len(training_data)
	print(f"\nTraining set accuracy: {final_acc:.2%}")

	return head_signature_vectors, head_entropies


def summarize_head_diversity(head_signature_vectors, head_entropies):
	print("\n" + "=" * 70)
	print("Head Comparison (Are heads learning unique patterns?)")
	print("=" * 70)

	head_signatures = []
	for h_idx in range(NUM_HEADS):
		# Flatten all last-row attentions across all examples.
		sig = np.concatenate(head_signature_vectors[h_idx], axis=0)
		head_signatures.append(sig)

		avg_ent = np.mean(head_entropies[h_idx])
		print(f"Head {h_idx + 1} average attention entropy: {avg_ent:.4f}")

	print("\nPairwise cosine similarity between head signatures:")
	for i in range(NUM_HEADS):
		for j in range(i + 1, NUM_HEADS):
			sim = cosine_similarity(head_signatures[i], head_signatures[j])
			print(f"  Head {i + 1} vs Head {j + 1}: {sim:.4f}")


def main():
	run_training()
	head_signature_vectors, head_entropies = evaluate_and_visualize()
	summarize_head_diversity(head_signature_vectors, head_entropies)


if __name__ == "__main__":
	main()
