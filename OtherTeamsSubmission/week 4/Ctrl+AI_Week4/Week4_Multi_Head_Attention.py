"""
Week 4: Multi-Head Attention
============================

Builds directly on Week 3's AttentionModel (Week3_Attention_Training_Improved.py).

Week 3 recap: scores = dot(raw_embedding, raw_embedding) -> softmax -> weighted sum.
One softmax over one set of raw embeddings = one head, no learned projections.

Week 4 adds two things Week 3 didn't have:
  1. Learned Q/K/V projections (per head) instead of comparing raw embeddings directly.
  2. Multiple heads, each with its own Q/K/V, run in parallel and combined
     via concatenation + a learned output projection (W_O).

See Week4_Multi_Head_Attention_explained.md for the conceptual walkthrough.

This file is structured to be run top-to-bottom, section by section:
  PART 1: Vocabulary + longer training sequences (from the exercise)
  PART 2: MultiHeadAttentionModel (num_heads is a constructor argument)
  PART 3: Sanity check -- num_heads=1 should train about as well as Week 3
  PART 4: Train num_heads=2 and num_heads=4 versions on the same data
  PART 5: Visualize + compare what each head attends to
  PART 6: Ablation experiments (zero one head / identical init) to test the
          exercise's "Questions to Think About"
"""

import numpy as np
from collections import defaultdict

np.random.seed(42)

# =============================================================================
# PART 1: VOCABULARY AND TRAINING SEQUENCES
# =============================================================================

vocabulary = [
    "Order", "Shipment", "Receive", "Restock",
    "Inventory", "Forecast", "Invoice", "Scenario",
]
token_to_id = {t: i for i, t in enumerate(vocabulary)}
id_to_token = {i: t for t, i in token_to_id.items()}
vocabulary_size = len(vocabulary)

# Longer sequences, as the exercise asks for (instead of Week 3's single-hop pairs)
training_sequences = [
    ["Order", "Shipment", "Receive", "Restock", "Inventory"],
    ["Shipment", "Receive", "Restock", "Inventory", "Forecast"],
    ["Receive", "Restock", "Inventory", "Forecast", "Order"],
    ["Inventory", "Forecast", "Scenario", "Order", "Shipment"],
]

embedding_dim = 8  # divides evenly by 1, 2, and 4 heads


# =============================================================================
# PART 2: MULTI-HEAD ATTENTION MODEL
# =============================================================================

def softmax(logits):
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def softmax_backward(weights, dweights):
    """Vector-Jacobian product for softmax: dscore_j = w_j * (dw_j - sum_k w_k*dw_k)"""
    dot = np.dot(weights, dweights)
    return weights * (dweights - dot)


class MultiHeadAttentionModel:
    """
    Self-attention with num_heads independent Q/K/V projections, combined via
    concatenation + a learned output projection, with a residual connection.

    num_heads=1 with head_dim=embedding_dim is the "single real self-attention
    head" upgrade from Week 3's raw-embedding dot product.
    """

    def __init__(self, vocab_size, emb_dim, num_heads, init_scale=0.2,
                 use_scaling=True, use_residual=True):
        assert emb_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        # Toggles below exist ONLY to reproduce the "naive first attempt" in
        # PART 7 (the improvement journey) -- the finished model always uses
        # use_scaling=True, use_residual=True. See Week4_Multi_Head_Attention_explained.md.
        self.use_scaling = use_scaling
        self.use_residual = use_residual

        self.E = np.random.randn(vocab_size, emb_dim) * 0.2

        # One W_Q, W_K, W_V per head, each projecting emb_dim -> head_dim
        scale = init_scale
        self.WQ = [np.random.randn(emb_dim, self.head_dim) * scale for _ in range(num_heads)]
        self.WK = [np.random.randn(emb_dim, self.head_dim) * scale for _ in range(num_heads)]
        self.WV = [np.random.randn(emb_dim, self.head_dim) * scale for _ in range(num_heads)]

        # Output projection: combines concatenated heads back into emb_dim
        self.WO = np.random.randn(emb_dim, emb_dim) * scale
        self.bO = np.zeros(emb_dim)

        # Final prediction layer (same role as Week 1-3's W_out/b_out)
        self.W_out = np.random.randn(emb_dim, vocab_size) * scale
        self.b_out = np.zeros(vocab_size)

    def forward_with_context(self, token_id, context_token_ids):
        """
        token_id: id of the current ("query") token
        context_token_ids: ids of tokens BEFORE the current one, in order

        Returns everything the backward pass needs, plus per-head attention
        weights for visualization.
        """
        x_q = self.E[token_id]
        context_embs = [self.E[t] for t in context_token_ids]

        head_outputs = []       # ctxvec_h per head, shape (head_dim,) each
        head_weights = []       # attention weights per head, for visualization
        head_cache = []         # (Q_h, K_h_list, V_h_list) per head, for backward

        for h in range(self.num_heads):
            Q_h = x_q @ self.WQ[h]

            if context_embs:
                K_h = [ctx @ self.WK[h] for ctx in context_embs]
                V_h = [ctx @ self.WV[h] for ctx in context_embs]
                scores = np.array([np.dot(Q_h, k) for k in K_h])
                if self.use_scaling:
                    scores = scores / np.sqrt(self.head_dim)
                weights = softmax(scores)
                ctxvec_h = np.zeros(self.head_dim)
                for w, v in zip(weights, V_h):
                    ctxvec_h += w * v
            else:
                K_h, V_h, weights = [], [], np.array([])
                ctxvec_h = np.zeros(self.head_dim)

            head_outputs.append(ctxvec_h)
            head_weights.append(weights)
            head_cache.append((Q_h, K_h, V_h))

        concat = np.concatenate(head_outputs)          # (emb_dim,)
        attn_out = concat @ self.WO + self.bO           # (emb_dim,)
        final_embedding = (x_q + attn_out) if self.use_residual else attn_out

        logits = final_embedding @ self.W_out + self.b_out
        probs = softmax(logits)

        cache = {
            "x_q": x_q,
            "context_embs": context_embs,
            "context_token_ids": context_token_ids,
            "head_cache": head_cache,
            "concat": concat,
            "final_embedding": final_embedding,
        }
        return probs, head_weights, cache

    def backward(self, token_id, true_id, probs, cache, learning_rate):
        x_q = cache["x_q"]
        context_embs = cache["context_embs"]
        context_token_ids = cache["context_token_ids"]
        head_cache = cache["head_cache"]
        concat = cache["concat"]
        final_embedding = cache["final_embedding"]

        # --- Output / prediction layer ---
        dlogits = probs.copy()
        dlogits[true_id] -= 1.0

        dW_out = np.outer(final_embedding, dlogits)
        db_out = dlogits
        dfinal = self.W_out @ dlogits  # (emb_dim,)

        # --- Residual split: dfinal flows into BOTH the skip path and attention path ---
        # (when use_residual=False there is no skip path, so no direct gradient into x_q here)
        d_xq_accum = dfinal.copy() if self.use_residual else np.zeros_like(dfinal)
        dattn_out = dfinal.copy()

        # --- Output projection W_O ---
        dWO = np.outer(concat, dattn_out)
        dbO = dattn_out
        dconcat = self.WO @ dattn_out   # (emb_dim,)

        dWQ = [np.zeros_like(w) for w in self.WQ]
        dWK = [np.zeros_like(w) for w in self.WK]
        dWV = [np.zeros_like(w) for w in self.WV]
        d_context_accum = defaultdict(lambda: np.zeros(self.emb_dim))

        # --- Backprop through each head independently ---
        for h in range(self.num_heads):
            Q_h, K_h, V_h = head_cache[h]
            start = h * self.head_dim
            dctxvec_h = dconcat[start:start + self.head_dim]

            if not context_embs:
                dQ_h = np.zeros(self.head_dim)
            else:
                scale_factor = (1.0 / np.sqrt(self.head_dim)) if self.use_scaling else 1.0
                scores = np.array([np.dot(Q_h, k) for k in K_h]) * scale_factor
                weights = softmax(scores)

                # d(ctxvec_h)/d(weights) and d(ctxvec_h)/d(V_h_j)
                dweights = np.array([np.dot(dctxvec_h, v) for v in V_h])
                dV_h = [w * dctxvec_h for w in weights]

                dscores = softmax_backward(weights, dweights) * scale_factor

                dQ_h = np.zeros(self.head_dim)
                dK_h = []
                for j in range(len(K_h)):
                    dQ_h += dscores[j] * K_h[j]
                    dK_h.append(dscores[j] * Q_h)

                for j, ctx_tok_id in enumerate(context_token_ids):
                    ctx_emb = context_embs[j]
                    dWK[h] += np.outer(ctx_emb, dK_h[j])
                    dWV[h] += np.outer(ctx_emb, dV_h[j])
                    d_context_accum[ctx_tok_id] += self.WK[h] @ dK_h[j]
                    d_context_accum[ctx_tok_id] += self.WV[h] @ dV_h[j]

            dWQ[h] += np.outer(x_q, dQ_h)
            d_xq_accum += self.WQ[h] @ dQ_h

        # --- Apply updates ---
        self.W_out -= learning_rate * dW_out
        self.b_out -= learning_rate * db_out
        self.WO -= learning_rate * dWO
        self.bO -= learning_rate * dbO
        for h in range(self.num_heads):
            self.WQ[h] -= learning_rate * dWQ[h]
            self.WK[h] -= learning_rate * dWK[h]
            self.WV[h] -= learning_rate * dWV[h]

        self.E[token_id] -= learning_rate * d_xq_accum
        for ctx_tok_id, grad in d_context_accum.items():
            self.E[ctx_tok_id] -= learning_rate * grad


# =============================================================================
# PART 3: TRAIN + EVALUATE HELPER
# =============================================================================

def train(model, sequences, epochs, learning_rate, log_every=None):
    history = defaultdict(list)
    for epoch in range(epochs):
        total_loss, total_correct, total_steps = 0.0, 0, 0
        for seq in sequences:
            seq_ids = [token_to_id[t] for t in seq]
            for i in range(len(seq_ids) - 1):
                current_id = seq_ids[i]
                target_id = seq_ids[i + 1]
                context_ids = seq_ids[:i]

                probs, _, cache = model.forward_with_context(current_id, context_ids)
                loss = -np.log(probs[target_id] + 1e-9)
                total_loss += loss
                total_steps += 1
                if np.argmax(probs) == target_id:
                    total_correct += 1

                model.backward(current_id, target_id, probs, cache, learning_rate)

        avg_loss = total_loss / total_steps
        accuracy = total_correct / total_steps
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)

        if log_every and (epoch + 1) % log_every == 0:
            print(f"  Epoch {epoch + 1:4d}  loss={avg_loss:.4f}  accuracy={accuracy:.1%}")

    return history


# =============================================================================
# PART 5: VISUALIZE + COMPARE WHAT EACH HEAD ATTENDS TO
# =============================================================================

def print_head_attention(model, seq):
    """
    Run the sequence forward once (no training) and print, for every head,
    a table of attention weights: rows = query position, columns = context
    tokens it could look back at.
    """
    seq_ids = [token_to_id[t] for t in seq]

    for i in range(1, len(seq_ids)):
        current_token = seq[i]
        context_ids = seq_ids[:i]
        context_tokens = seq[:i]

        # Query with token i, attending over everything before it (0..i-1)
        _, head_weights, _ = model.forward_with_context(seq_ids[i], context_ids)

        print(f"  Query token: '{current_token}' (position {i}) attending over {context_tokens}")
        for h, weights in enumerate(head_weights):
            weight_str = "  ".join(f"{tok}={w:.1%}" for tok, w in zip(context_tokens, weights))
            top_idx = int(np.argmax(weights))
            print(f"    Head {h + 1}: {weight_str}   -> top: {context_tokens[top_idx]}")
        print()


# =============================================================================
# PART 6: ABLATION EXPERIMENTS -- test the exercise's reasoning questions
# =============================================================================

def evaluate_accuracy(model, sequences):
    correct, total = 0, 0
    for seq in sequences:
        seq_ids = [token_to_id[t] for t in seq]
        for i in range(len(seq_ids) - 1):
            probs, _, _ = model.forward_with_context(seq_ids[i], seq_ids[:i])
            if np.argmax(probs) == seq_ids[i + 1]:
                correct += 1
            total += 1
    return correct / total


def zero_head_ablation(model, sequences):
    """Zero one head's contribution to W_O (its rows), one head at a time,
    and measure accuracy with that head silenced -- tests: 'if one head
    disappears, does the model still work?'"""
    results = {}
    original_WO = model.WO.copy()
    for h in range(model.num_heads):
        start = h * model.head_dim
        model.WO[start:start + model.head_dim, :] = 0.0
        results[h] = evaluate_accuracy(model, sequences)
        model.WO = original_WO.copy()
    return results


def identical_init_divergence_test(vocab_size, emb_dim, num_heads, sequences, epochs, learning_rate):
    """Force head 0 and head 1 to start with IDENTICAL Q/K/V weights, then
    train normally -- tests: 'if every head learns the same thing, have we
    gained anything?' by seeing whether training pulls them apart or leaves
    them stuck as duplicates."""
    model = MultiHeadAttentionModel(vocab_size, emb_dim, num_heads)
    model.WQ[1] = model.WQ[0].copy()
    model.WK[1] = model.WK[0].copy()
    model.WV[1] = model.WV[0].copy()

    initial_diff = 0.0  # identical at init, by construction

    train(model, sequences, epochs=epochs, learning_rate=learning_rate)

    final_diff_WQ = np.linalg.norm(model.WQ[0] - model.WQ[1])
    final_diff_WK = np.linalg.norm(model.WK[0] - model.WK[1])
    final_diff_WV = np.linalg.norm(model.WV[0] - model.WV[1])
    return model, initial_diff, (final_diff_WQ, final_diff_WK, final_diff_WV)


if __name__ == "__main__":
    print("=" * 80)
    print("PART 3: SANITY CHECK -- num_heads=1 should match Week 3's accuracy range")
    print("=" * 80)
    print()

    model_1head = MultiHeadAttentionModel(vocabulary_size, embedding_dim, num_heads=1)
    history_1head = train(model_1head, training_sequences, epochs=300, learning_rate=0.05, log_every=60)

    print()
    print(f"Final: loss={history_1head['loss'][-1]:.4f}  accuracy={history_1head['accuracy'][-1]:.1%}")
    print("(Week 3's improved model reached 80-100% accuracy on its simpler single-hop pairs --")
    print(" this is a harder task: longer sequences, more context choices per step.)")
    print()

    print("=" * 80)
    print("PART 4: MULTI-HEAD MODELS (2 heads, then 4 heads) ON THE SAME DATA")
    print("=" * 80)
    print()

    print("-- Training 2-head model --")
    model_2head = MultiHeadAttentionModel(vocabulary_size, embedding_dim, num_heads=2)
    history_2head = train(model_2head, training_sequences, epochs=300, learning_rate=0.05, log_every=60)
    print(f"Final: loss={history_2head['loss'][-1]:.4f}  accuracy={history_2head['accuracy'][-1]:.1%}")
    print()

    print("-- Training 4-head model --")
    model_4head = MultiHeadAttentionModel(vocabulary_size, embedding_dim, num_heads=4)
    history_4head = train(model_4head, training_sequences, epochs=300, learning_rate=0.05, log_every=60)
    print(f"Final: loss={history_4head['loss'][-1]:.4f}  accuracy={history_4head['accuracy'][-1]:.1%}")
    print()

    print("Summary (final epoch):")
    print(f"  1 head : loss={history_1head['loss'][-1]:.4f}  accuracy={history_1head['accuracy'][-1]:.1%}")
    print(f"  2 heads: loss={history_2head['loss'][-1]:.4f}  accuracy={history_2head['accuracy'][-1]:.1%}")
    print(f"  4 heads: loss={history_4head['loss'][-1]:.4f}  accuracy={history_4head['accuracy'][-1]:.1%}")
    print()

    print("=" * 80)
    print("PART 5: WHAT DOES EACH HEAD ATTEND TO?")
    print("=" * 80)
    print()

    demo_seq = ["Inventory", "Forecast", "Scenario", "Order", "Shipment"]
    print(f"Demo sequence: {' -> '.join(demo_seq)}")
    print()

    print("--- 2-head model ---")
    print_head_attention(model_2head, demo_seq)

    print("--- 4-head model ---")
    print_head_attention(model_4head, demo_seq)

    # -------------------------------------------------------------------
    # PART 5b: THE REAL TEST -- an ambiguous case a no-context model can't solve
    # -------------------------------------------------------------------
    print("=" * 80)
    print("PART 5b: 'Forecast' IS GENUINELY AMBIGUOUS -- DOES CONTEXT RESOLVE IT?")
    print("=" * 80)
    print()
    print("In the training data, 'Forecast' is immediately preceded by 'Inventory'")
    print("in BOTH of these cases, yet the correct next word is different:")
    print()
    print("  Case A -- seq3: Receive -> Restock -> Inventory -> Forecast -> [Order]")
    print("  Case B -- seq4:                       Inventory -> Forecast -> [Scenario]")
    print()
    print("A model with NO context (Week 1/2 style: predict from current word alone)")
    print("cannot tell these apart -- both start from the same 'Forecast' embedding,")
    print("so it must give the SAME prediction to both, and will be wrong on one.")
    print("An attention model can look further back and tell the two cases apart.")
    print()

    case_a_context = [token_to_id[t] for t in ["Receive", "Restock", "Inventory"]]
    case_b_context = [token_to_id[t] for t in ["Inventory"]]
    forecast_id = token_to_id["Forecast"]

    for label, model in [("1 head", model_1head), ("2 heads", model_2head), ("4 heads", model_4head)]:
        probs_a, _, _ = model.forward_with_context(forecast_id, case_a_context)
        probs_b, _, _ = model.forward_with_context(forecast_id, case_b_context)
        pred_a = id_to_token[int(np.argmax(probs_a))]
        pred_b = id_to_token[int(np.argmax(probs_b))]
        print(f"  {label:8s}: Case A -> predicts '{pred_a}' (want Order)   "
              f"Case B -> predicts '{pred_b}' (want Scenario)")
    print()
    print("No-context baseline (just embedding @ W_out, ignoring all history) would have")
    print("to predict the SAME word for both cases -- it literally cannot see the difference.")
    print("If the attention models above printed different, correct words for A and B,")
    print("that's context actually being used, not just memorization capacity.")
    print()

    print("=" * 80)
    print("PART 6: ABLATION EXPERIMENTS")
    print("=" * 80)
    print()

    print("-- Experiment 1: zero out one head at a time (4-head model) --")
    baseline_acc = evaluate_accuracy(model_4head, training_sequences)
    print(f"  Baseline accuracy (all 4 heads active): {baseline_acc:.1%}")
    ablation_results = zero_head_ablation(model_4head, training_sequences)
    for h, acc in ablation_results.items():
        drop = baseline_acc - acc
        print(f"  Silence Head {h + 1}: accuracy={acc:.1%}  (drop of {drop:.1%})")
    print()
    print("If accuracy barely drops when a head is silenced, that head's job was")
    print("redundant with another head's. A big drop means that head was carrying")
    print("information no other head had learned to provide.")
    print()

    print("-- Experiment 2: force two heads to start IDENTICAL, then train --")
    model_twin, initial_diff, final_diffs = identical_init_divergence_test(
        vocabulary_size, embedding_dim, num_heads=2,
        sequences=training_sequences, epochs=300, learning_rate=0.05,
    )
    print(f"  Head 1 vs Head 2 weight difference BEFORE training: {initial_diff:.4f} (identical by construction)")
    print(f"  Head 1 vs Head 2 weight difference AFTER training:")
    print(f"    ||WQ1 - WQ2|| = {final_diffs[0]:.4f}")
    print(f"    ||WK1 - WK2|| = {final_diffs[1]:.4f}")
    print(f"    ||WV1 - WV2|| = {final_diffs[2]:.4f}")
    print()
    if sum(final_diffs) > 0.01:
        print("  -> The heads DIVERGED even though they started identical. Gradient descent")
        print("     had no reason to keep them tied together (each feeds a different slice")
        print("     of W_O), so symmetry broke and they drifted toward different behavior.")
    else:
        print("  -> The heads stayed nearly identical -- in this run, nothing pushed them apart.")
    print()

    # =========================================================================
    # PART 7: THE IMPROVEMENT JOURNEY -- from a naive first attempt to 100%
    #
    # Same idea as Week3_Attention_Training_Improved.py's "v1 problems -> v2
    # fixes" writeup, but for multi-head attention. Every number below is
    # from an actual training run (use_scaling / use_residual / init_scale
    # are constructor toggles that exist ONLY to reproduce these mistakes --
    # the finished model always uses use_scaling=True, use_residual=True).
    # =========================================================================
    print("=" * 80)
    print("PART 7: THE IMPROVEMENT JOURNEY -- NAIVE FIRST ATTEMPT TO 100% ACCURACY")
    print("=" * 80)
    print()

    def run_variant(init_scale, use_scaling, use_residual, lr, epochs=100):
        np.random.seed(42)
        m = MultiHeadAttentionModel(vocabulary_size, embedding_dim, num_heads=4,
                                     init_scale=init_scale, use_scaling=use_scaling,
                                     use_residual=use_residual)
        h = train(m, training_sequences, epochs=epochs, learning_rate=lr)
        return h

    print("-- Step 1: the naive first attempt --")
    print("   Reused Week 3's learning rate (0.3) and a larger init scale (0.8),")
    print("   with no scaled dot-product and no residual connection:")
    h_naive = run_variant(init_scale=0.8, use_scaling=False, use_residual=False, lr=0.3)
    naive_loss = h_naive["loss"][-1]
    print(f"     final loss = {'NaN (diverged)' if np.isnan(naive_loss) else f'{naive_loss:.4f}'}"
          f"   final accuracy = {h_naive['accuracy'][-1]:.1%}"
          f"   (random guessing on {vocabulary_size} words = {1/vocabulary_size:.1%})")
    print("   -> The loss blows up to NaN. The model never learns anything --")
    print("      it's stuck at the random-guess baseline.")
    print()

    print("-- Step 2: which single fix actually stops the divergence? --")
    print("   Testing each candidate fix ALONE, everything else still naive (lr=0.3):")
    for label, kwargs in [
        ("only: sane init (0.8 -> 0.2)", dict(init_scale=0.2, use_scaling=False, use_residual=False)),
        ("only: scaled dot-product (add /sqrt(head_dim))", dict(init_scale=0.8, use_scaling=True, use_residual=False)),
        ("only: residual connection", dict(init_scale=0.8, use_scaling=False, use_residual=True)),
        ("only: lower learning rate (0.3 -> 0.05)", dict(init_scale=0.8, use_scaling=False, use_residual=False, lr=0.05)),
    ]:
        lr = kwargs.pop("lr", 0.3)
        h = run_variant(lr=lr, **kwargs)
        loss = h["loss"][-1]
        status = "still diverges (NaN)" if np.isnan(loss) else f"stable, loss={loss:.4f}"
        print(f"     {label:48s} -> {status}, accuracy={h['accuracy'][-1]:.1%}")
    print()
    print("   -> Init scale, scaling, and residual EACH FAIL to stop the divergence")
    print("      on their own. Only the learning rate was actually causing it -- with")
    print("      4 heads x 3 projections + an output projection all updating jointly,")
    print("      Week 3's learning rate was simply too aggressive for this many")
    print("      interacting weight matrices. (Week 3 hit this same wall going from")
    print("      Week 1/2's single embedding to one attention head -- see its 0.1 -> 0.01")
    print("      fix. More moving parts need an even smaller learning rate.)")
    print()

    print("-- Step 3: with the learning rate fixed (0.05), what do scaling/residual add? --")
    grid = [
        ("A. neither scaling nor residual", dict(init_scale=0.8, use_scaling=False, use_residual=False)),
        ("B. + residual connection", dict(init_scale=0.8, use_scaling=False, use_residual=True)),
        ("C. + scaled dot-product only", dict(init_scale=0.8, use_scaling=True, use_residual=False)),
        ("D. + both", dict(init_scale=0.8, use_scaling=True, use_residual=True)),
        ("E. + sane init too [FINAL CONFIG]", dict(init_scale=0.2, use_scaling=True, use_residual=True)),
    ]
    for label, kwargs in grid:
        h = run_variant(lr=0.05, **kwargs)
        accs = np.array(h["accuracy"])
        epoch_100 = next((i + 1 for i, a in enumerate(accs) if a == 1.0), None)
        epoch_100_str = f"epoch {epoch_100}" if epoch_100 else "never in 100 epochs"
        print(f"     {label:38s} -> final acc={accs[-1]:6.1%}  loss={h['loss'][-1]:.4f}  "
              f"reached 100% at {epoch_100_str}")
    print()
    print("   -> Once the learning rate is sane, the RESIDUAL CONNECTION is what takes")
    print("      accuracy from a 75% ceiling to 100% (by epoch 7). The scaled dot-product")
    print("      alone, without the residual, made no measurable difference here --")
    print("      head_dim is tiny (2) in this toy setup, so dot products never got large")
    print("      enough to need rescaling. It still belongs in the implementation for")
    print("      correctness (larger head_dim or longer sequences WILL need it), but on")
    print("      this dataset the residual connection was the fix that actually mattered.")
    print()
    print("   -> A larger init scale (0.8) even reached 100% faster (epoch 7) than the")
    print("      smaller default (0.2, epoch 16) once the other fixes were in place --")
    print("      a reminder that 'smaller init is always safer' isn't a universal rule,")
    print("      it depends on the rest of the configuration.")
    print()

    print("-- Step 4: more epochs mainly polishes an already-solved task --")
    h_100 = run_variant(init_scale=0.2, use_scaling=True, use_residual=True, lr=0.05, epochs=100)
    h_300 = run_variant(init_scale=0.2, use_scaling=True, use_residual=True, lr=0.05, epochs=300)
    print(f"     100 epochs -> loss={h_100['loss'][-1]:.4f}  accuracy={h_100['accuracy'][-1]:.1%}")
    print(f"     300 epochs -> loss={h_300['loss'][-1]:.4f}  accuracy={h_300['accuracy'][-1]:.1%}")
    print("   -> Same 100% accuracy either way -- extra epochs here just shrink the loss")
    print("      further, the same diminishing-returns pattern Week 3 saw once a")
    print("      deterministic toy task is already solved.")
    print()

    print("=" * 80)
    print("SUMMARY: LOW -> HIGH ACCURACY, IN ORDER OF ACTUAL IMPACT")
    print("=" * 80)
    print("""
  1. Learning rate (0.3 -> 0.05)   Fixes catastrophic NaN divergence. THE critical fix --
                                    nothing else mattered until this was fixed.
  2. Residual connection            Takes accuracy from a 75% ceiling to 100%, and fast
                                    (converges by epoch 7 once added).
  3. Scaled dot-product (/sqrt(d))  Textbook-correct and cheap to keep, but made no
                                    measurable difference on this tiny head_dim=2 setup.
  4. Sane init scale (0.8 -> 0.2)   Not required for correctness here -- in this run the
                                    larger init even converged faster. Worth using anyway
                                    for robustness on other datasets/vocab sizes.
  5. More epochs (100 -> 300)       Diminishing returns once accuracy is already 100% --
                                    only the loss keeps shrinking.
""")
    print()
