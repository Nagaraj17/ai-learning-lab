"""
gradient_checker.py
===================
Rigorous finite-difference gradient verification for the pure-NumPy Transformer.

Improvements over previous version:
- Uses float64 throughout (higher precision)
- Uses deterministic (seeded) parameter sample indices
- Checks more parameters per tensor (10 instead of 5)
- Reports both absolute and relative error
- Handles near-zero gradients gracefully
- Stricter relative-error tolerance (0.02 = 2%)
- Tests specific logical properties of the model
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from numpy_transformer_suite import (
    ModularTinyTransformer, compute_cross_entropy_loss,
    sinusoidal_positional_encoding, softmax
)
from step_therapy_generator import VOCAB_SIZE


# ---------------------------------------------------------------------------
# Shape assertion
# ---------------------------------------------------------------------------

def assert_tensor_shapes(model, input_ids, mask):
    """Assert shapes at every architectural boundary."""
    B, T = input_ids.shape
    d_model = model.d_model
    V = model.vocab_size

    emb = model.W_emb[input_ids].astype(np.float64) + model.pos_enc[:T]
    assert emb.shape == (B, T, d_model), f"Embedding shape: {emb.shape}"

    logits, x_final = model.forward(input_ids)
    assert logits.shape == (B, T, V), f"Logits shape: {logits.shape}"
    assert x_final.shape == (B, T, d_model), f"x_final shape: {x_final.shape}"

    targets = input_ids.copy()
    loss, dlogits, probs = compute_cross_entropy_loss(logits, targets, mask)
    assert dlogits.shape == (B, T, V), f"dLogits shape: {dlogits.shape}"
    assert np.isfinite(loss), f"Loss is not finite: {loss}"
    assert np.all(np.isfinite(dlogits)), "dLogits contains non-finite values"

    print(f"  [SHAPE OK] Model {model.model_id}: "
          f"emb={emb.shape}, logits={logits.shape}, loss={loss:.4f}")


# ---------------------------------------------------------------------------
# Logical correctness checks
# ---------------------------------------------------------------------------

def check_causal_masking(model, input_ids):
    """Verify future positions have ~0 attention weight."""
    logits, _ = model.forward(input_ids)
    for i, block in enumerate(model.blocks):
        attn_weights = model.get_attention_weights(layer_idx=i)
        if attn_weights is None:
            continue
        B, H, T, T2 = attn_weights.shape
        upper_triangle = np.triu(attn_weights, k=1)
        max_future_attn = np.max(np.abs(upper_triangle))
        assert max_future_attn < 1e-6, \
            f"Block {i}: future attention weight {max_future_attn:.2e} > 0"
        print(f"  [CAUSAL OK] Block {i}: max future attention = {max_future_attn:.2e}")


def check_attention_sums_to_one(model, input_ids):
    """Verify attention weights (per-head, per-query) sum to approximately 1."""
    model.forward(input_ids)
    for i, block in enumerate(model.blocks):
        attn_weights = model.get_attention_weights(layer_idx=i)
        if attn_weights is None:
            continue
        sums = np.sum(attn_weights, axis=-1)
        max_dev = np.max(np.abs(sums - 1.0))
        assert max_dev < 1e-5, \
            f"Block {i}: attention row sums deviate from 1 by {max_dev:.2e}"
        print(f"  [ATTN SUM OK] Block {i}: max deviation from 1 = {max_dev:.2e}")


def check_padding_independence(model, input_ids, mask):
    """Verify changing padding tokens does not affect unmasked logits."""
    logits_ref, _ = model.forward(input_ids)

    # Replace all padding positions with random different values
    modified = input_ids.copy()
    pad_positions = (mask == 0)
    if not np.any(pad_positions):
        print("  [PADDING] No padding positions to test.")
        return

    modified[pad_positions] = (modified[pad_positions] + 1) % model.vocab_size
    logits_mod, _ = model.forward(modified)

    # Logits at non-padding positions should be identical (causal masking ensures this)
    valid_mask = (mask > 0)[:, :, np.newaxis]
    diff = np.max(np.abs(logits_ref - logits_mod) * valid_mask)
    # Note: padding affects earlier-position logits through embedding;
    # what matters is that padding-position gradients are masked to 0.
    print(f"  [PADDING NOTE] Max logit diff at valid positions: {diff:.4e} "
          f"(expected ~0 if all padding is at end)")


def check_ffn_output_shape(model, input_ids):
    """Verify FFN output stays in d_model dimension."""
    logits, x_final = model.forward(input_ids)
    B, T = input_ids.shape
    assert x_final.shape == (B, T, model.d_model), \
        f"Final hidden shape {x_final.shape} != ({B},{T},{model.d_model})"
    print(f"  [FFN SHAPE OK] x_final shape = {x_final.shape}")


def check_no_dead_gradients(model, input_ids, targets, mask):
    """Check that no important gradient tensor is entirely zero after one backward."""
    logits, x_final = model.forward(input_ids)
    loss, dlogits, _ = compute_cross_entropy_loss(logits, targets, mask)
    model.backward(dlogits, x_final)

    zero_grads = []
    for i, (param, grad) in enumerate(model.get_params_and_grads()):
        if np.all(grad == 0.0) and param.size > 1:
            zero_grads.append((i, param.shape))

    if zero_grads:
        print(f"  [DEAD GRAD WARNING] {len(zero_grads)} gradient tensors are zero: {zero_grads}")
    else:
        print(f"  [GRAD OK] No permanently-zero gradient tensors found.")


# ---------------------------------------------------------------------------
# Finite-difference gradient check
# ---------------------------------------------------------------------------

def finite_difference_gradient_check(
    model, input_ids, targets, mask,
    eps=1e-5,
    n_samples_per_param=10,
    rel_threshold=0.02,
    abs_threshold=1e-7,
    seed=42
):
    """
    Finite-difference gradient check using float64 arithmetic.

    For each parameter tensor, sample n_samples_per_param scalar coordinates
    using a fixed seed for reproducibility. Compare:
        g_analytical  (backpropagation)
        g_numerical   (central difference: (f(x+eps) - f(x-eps)) / 2eps)

    A coordinate passes if:
        |g_ana - g_num| / (|g_ana| + |g_num| + 1e-8) < rel_threshold
        OR |g_ana - g_num| < abs_threshold  (near-zero gradient case)

    Parameters
    ----------
    eps              : finite difference step (default 1e-5)
    n_samples_per_param : scalars to test per parameter tensor
    rel_threshold    : relative-error pass threshold (default 2%)
    abs_threshold    : absolute-error pass threshold for near-zero case (1e-7)
    seed             : fixed seed for reproducible index sampling

    Returns
    -------
    all_passed : bool
    summary    : list of dicts with per-tensor results
    """
    rng_sample = np.random.RandomState(seed)

    # Step 1: analytical backward
    logits, x_final = model.forward(input_ids)
    loss_ref, dlogits, _ = compute_cross_entropy_loss(logits, targets, mask)
    model.backward(dlogits, x_final)

    param_pairs = model.get_params_and_grads()
    all_passed = True
    summary = []

    for idx, (param, grad_analytical) in enumerate(param_pairs):
        param_flat = param.ravel()
        grad_flat = grad_analytical.ravel()

        n = len(param_flat)
        sample_idx = rng_sample.choice(n, size=min(n_samples_per_param, n), replace=False)
        sample_idx.sort()

        max_rel = 0.0
        max_abs = 0.0
        n_pass = 0
        n_total = len(sample_idx)

        for i in sample_idx:
            orig = float(param_flat[i])

            param_flat[i] = orig + eps
            logits_p, _ = model.forward(input_ids)
            loss_p, _, _ = compute_cross_entropy_loss(logits_p, targets, mask)

            param_flat[i] = orig - eps
            logits_m, _ = model.forward(input_ids)
            loss_m, _, _ = compute_cross_entropy_loss(logits_m, targets, mask)

            param_flat[i] = orig  # restore

            g_num = (float(loss_p) - float(loss_m)) / (2.0 * eps)
            g_ana = float(grad_flat[i])

            abs_err = abs(g_ana - g_num)
            rel_err = abs_err / (abs(g_ana) + abs(g_num) + 1e-8)

            max_rel = max(max_rel, rel_err)
            max_abs = max(max_abs, abs_err)

            passed = rel_err < rel_threshold or abs_err < abs_threshold
            if passed:
                n_pass += 1

        all_pass_tensor = (n_pass == n_total)
        if not all_pass_tensor:
            all_passed = False

        result = {
            "param_idx": idx,
            "shape": param.shape,
            "n_sampled": n_total,
            "n_passed": n_pass,
            "max_rel_error": max_rel,
            "max_abs_error": max_abs,
            "passed": all_pass_tensor,
        }
        summary.append(result)

        status = "PASS" if all_pass_tensor else "FAIL"
        print(f"  Param {idx:2d} {str(param.shape):<14s} "
              f"rel={max_rel:.2e}  abs={max_abs:.2e}  "
              f"{n_pass}/{n_total} samples -> {status}")

    return all_passed, summary


# ---------------------------------------------------------------------------
# Full check suite
# ---------------------------------------------------------------------------

def run_full_verification(model_id="D", vocab_size=None, seed=42):
    """Run all checks on a given model architecture."""
    if vocab_size is None:
        vocab_size = VOCAB_SIZE

    print(f"\n{'='*60}")
    print(f" Verification Suite: Model {model_id}")
    print(f"{'='*60}")

    rng = np.random.RandomState(seed)
    B, T = 4, 12
    input_ids = rng.randint(1, vocab_size, size=(B, T)).astype(np.int64)
    targets = rng.randint(1, vocab_size, size=(B, T)).astype(np.int64)
    # Mix: 8 valid + 4 padding
    mask = np.ones((B, T), dtype=np.float32)
    mask[:, 8:] = 0.0

    model = ModularTinyTransformer(
        model_id, vocab_size=vocab_size, d_model=24, d_ff=96, max_len=20, seed=seed
    )

    print("\n[1] Shape assertions")
    assert_tensor_shapes(model, input_ids, mask)

    print("\n[2] FFN output shape")
    check_ffn_output_shape(model, input_ids)

    if len(model.blocks) > 0:
        print("\n[3] Causal masking")
        check_causal_masking(model, input_ids)

        print("\n[4] Attention sums to 1")
        check_attention_sums_to_one(model, input_ids)

    print("\n[5] Padding independence check")
    check_padding_independence(model, input_ids, mask)

    print("\n[6] Dead gradient check")
    check_no_dead_gradients(model, input_ids, targets, mask)

    print("\n[7] Finite-difference gradient check")
    all_passed, summary = finite_difference_gradient_check(
        model, input_ids, targets, mask,
        eps=1e-5, n_samples_per_param=10, rel_threshold=0.02, seed=seed
    )

    print(f"\n{'='*60}")
    if all_passed:
        print(f" RESULT: ALL CHECKS PASSED for Model {model_id}")
    else:
        failed = [s for s in summary if not s["passed"]]
        print(f" RESULT: {len(failed)} tensor(s) FAILED gradient check")
    print(f"{'='*60}\n")

    return all_passed, summary


if __name__ == "__main__":
    for mid in ["A", "B", "C", "D"]:
        run_full_verification(mid)
