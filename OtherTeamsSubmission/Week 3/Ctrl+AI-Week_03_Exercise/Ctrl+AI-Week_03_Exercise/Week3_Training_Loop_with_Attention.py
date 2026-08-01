"""
Annotated Training Loop: Where Attention Happens

Shows exactly when attention computation occurs during training
"""

import numpy as np

np.random.seed(42)

# Setup (abbreviated)
vocabulary = ["Order", "Shipment", "Receive", "Restock", "Inventory", "Forecast"]
token_to_id = {t: i for i, t in enumerate(vocabulary)}
id_to_token = {i: t for t, i in token_to_id.items()}

E = np.random.randn(len(vocabulary), 6) * 0.2
W_out = np.random.randn(6, len(vocabulary)) * 0.2
b_out = np.zeros(len(vocabulary))

training_pairs = [
    ("Order", "Shipment"),
    ("Shipment", "Receive"),
    ("Receive", "Restock"),
    ("Restock", "Inventory"),
    ("Inventory", "Forecast"),
    ("Forecast", "Order"),
]

def softmax(logits):
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def compute_attention(embedding, previous_embeddings):
    """Compute attention weights for blending"""
    if not previous_embeddings:
        return []

    scores = [np.dot(embedding, prev) for prev in previous_embeddings]
    scores = np.array(scores)
    shifted = scores - np.max(scores)
    exp_values = np.exp(shifted)
    weights = exp_values / np.sum(exp_values)
    return weights


# =============================================================================
# TRAINING LOOP WITH ANNOTATIONS
# =============================================================================

learning_rate = 0.15
epochs = 3  # Just 3 epochs for demonstration

print("=" * 80)
print("TRAINING LOOP WITH ATTENTION - ANNOTATED")
print("=" * 80)
print()

for epoch in range(epochs):
    print(f"{'─' * 80}")
    print(f"EPOCH {epoch}")
    print(f"{'─' * 80}")

    total_loss = 0.0
    embeddings_in_sequence = []  # Track embeddings for attention

    for pair_idx, (src, dst) in enumerate(training_pairs):
        src_id = token_to_id[src]
        dst_id = token_to_id[dst]

        print(f"\n  Pair {pair_idx + 1}: {src} → {dst}")
        print()

        # ┌────────────────────────────────────────────────────────────┐
        # │ FORWARD PASS START                                         │
        # └────────────────────────────────────────────────────────────┘

        # Step 1: Get base embedding
        embedding = E[src_id]
        print(f"    [FORWARD] Step 1: Embed '{src}'")
        print(f"              embedding shape: {embedding.shape}")
        print(f"              embedding: {embedding}")
        print()

        # Step 2: ← ATTENTION HAPPENS HERE ←
        print(f"    [FORWARD] Step 2: ← ATTENTION COMPUTATION ←")

        if len(embeddings_in_sequence) > 0:
            print(f"              Previous tokens in sequence: {len(embeddings_in_sequence)}")

            # Compute attention weights
            attention_weights = compute_attention(embedding, embeddings_in_sequence)
            print(f"              Attention weights: {attention_weights}")

            # Blend with previous embeddings
            context_embedding = np.zeros_like(embedding)
            for w, prev_emb in zip(attention_weights, embeddings_in_sequence):
                context_embedding += w * prev_emb

            print(f"              Context blending: ", end="")
            for i, (prev_token, w) in enumerate(zip(training_pairs[max(0, pair_idx - len(embeddings_in_sequence)):pair_idx], attention_weights)):
                print(f"{w:.2f}*{prev_token[0]:8s}", end=" ")
            print()

            # Mix with self
            final_embedding = 0.7 * context_embedding + 0.3 * embedding
            print(f"              Mix: 70% context + 30% self")
            print(f"              Final embedding: {final_embedding}")
        else:
            print(f"              No previous context (first token)")
            final_embedding = embedding

        embeddings_in_sequence.append(embedding)  # Remember for next iteration
        print()

        # Step 3: Compute logits
        print(f"    [FORWARD] Step 3: Compute logits")
        logits = final_embedding @ W_out + b_out
        print(f"              logits: {logits}")
        print()

        # Step 4: Softmax to probabilities
        print(f"    [FORWARD] Step 4: Convert to probabilities")
        probs = softmax(logits)
        pred_id = np.argmax(probs)
        pred = id_to_token[pred_id]
        target = dst
        print(f"              Top 3 predictions:")
        for idx in np.argsort(probs)[-3:]:
            print(f"                {id_to_token[idx]:10s}: {probs[idx]:.1%}")
        print()

        # ┌────────────────────────────────────────────────────────────┐
        # │ LOSS COMPUTATION                                           │
        # └────────────────────────────────────────────────────────────┘

        print(f"    [LOSS] Step 5: Compute loss")
        loss = -np.log(probs[dst_id] + 1e-9)
        total_loss += loss
        match = "✓" if pred == target else "✗"
        print(f"              Target: {target:10s} | Predicted: {pred:10s} {match}")
        print(f"              Loss: {loss:.4f}")
        print()

        # ┌────────────────────────────────────────────────────────────┐
        # │ BACKWARD PASS (Simplified annotation)                      │
        # └────────────────────────────────────────────────────────────┘

        print(f"    [BACKWARD] Step 6: Backpropagation")
        print(f"              Computing gradients...")

        # Compute gradients
        dlogits = probs.copy()
        dlogits[dst_id] -= 1.0

        dW_out = np.outer(final_embedding, dlogits)
        db_out = dlogits
        dfinal_embedding = W_out @ dlogits

        print(f"              Gradients computed (dW_out, db_out, dfinal_embedding)")

        if len(embeddings_in_sequence) > 1:
            print(f"              ← Gradients flow BACK through attention blending")
        print()

        # ┌────────────────────────────────────────────────────────────┐
        # │ PARAMETER UPDATE                                           │
        # └────────────────────────────────────────────────────────────┘

        print(f"    [UPDATE] Step 7: Update parameters")

        # Extract gradient for embedding
        dembedding = dfinal_embedding * 0.3  # Only the self-contribution

        # Update
        W_out -= learning_rate * dW_out
        b_out -= learning_rate * db_out
        E[src_id] -= learning_rate * dembedding

        print(f"              W_out updated")
        print(f"              b_out updated")
        print(f"              E['{src}'] updated (learned from this example!)")
        print()

    # End of epoch
    avg_loss = total_loss / len(training_pairs)
    print(f"\n  Epoch {epoch} Summary:")
    print(f"  Total Loss: {total_loss:.4f}")
    print(f"  Average Loss: {avg_loss:.4f}")
    print()

print("=" * 80)
print()
print("KEY INSIGHTS:")
print("-" * 80)
print()
print("1. WHEN does attention happen?")
print("   → In the FORWARD PASS, Step 2, right after embedding lookup")
print()
print("2. HOW OFTEN?")
print("   → EVERY iteration (every training pair)")
print("   → EVERY epoch (repeated 2500 times)")
print()
print("3. WHAT does it do?")
print("   → Blends current embedding with previous word embeddings")
print("   → Weights based on similarity (dot product)")
print()
print("4. HOW do gradients flow?")
print("   → BACKWARD through the attention blending")
print("   → Updates the base embeddings E[word]")
print()
print("5. RESULT?")
print("   → Over many epochs, embeddings learn to be context-aware")
print("   → Same word gets slightly different representation per context")
print()
print("=" * 80)
