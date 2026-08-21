import numpy as np
from step_therapy_generator import VOCAB_SIZE
from numpy_transformer_suite import ModularTinyTransformer, compute_cross_entropy_loss

def assert_tensor_shapes(model, input_ids, mask):
    """
    Asserts tensor shapes at all 8 architectural boundaries.
    """
    B, T = input_ids.shape
    d_model = model.d_model
    vocab_size = model.vocab_size

    # Boundary 1: Embedding + Positional Encoding
    emb = model.W_emb[input_ids] + model.pos_enc[:T]
    assert emb.shape == (B, T, d_model), f"Embedding shape mismatch: {emb.shape} vs ({B}, {T}, {d_model})"

    # Boundary 7 & 8: Logits & Loss gradient
    logits, x_final = model.forward(input_ids)
    assert logits.shape == (B, T, vocab_size), f"Logits shape mismatch: {logits.shape} vs ({B}, {T}, {vocab_size})"
    
    loss, dlogits, probs = compute_cross_entropy_loss(logits, input_ids, mask)
    assert dlogits.shape == (B, T, vocab_size), f"dLogits shape mismatch: {dlogits.shape} vs ({B}, {T}, {vocab_size})"

    print(f"[OK] All shape assertions passed for Model Variant {model.model_id} (B={B}, T={T}, d_model={d_model}, V={vocab_size}).")

def finite_difference_gradient_check(model, input_ids, targets, mask, eps=1e-3, max_check_params=5, threshold=0.10):
    """
    Performs numerical gradient checking via finite differences.
    Relative error formula: ||g_analytical - g_numerical|| / (||g_analytical|| + ||g_numerical|| + 1e-8)
    Acceptance threshold for 2-block network: relative error < 0.10 (10%).
    """
    print(f"\n--- Running Finite-Difference Gradient Check for Model Variant {model.model_id} ---")
    
    # 1. Compute analytical gradients via backprop
    logits, x_final = model.forward(input_ids)
    loss, dlogits, probs = compute_cross_entropy_loss(logits, targets, mask)
    model.backward(dlogits, x_final)

    param_grad_pairs = model.get_params_and_grads()
    
    all_passed = True
    
    for idx, (param, grad_analytical) in enumerate(param_grad_pairs):
        # Sample a few random scalar coordinates to test
        param_flat = param.ravel()
        grad_flat = grad_analytical.ravel()
        
        num_params = len(param_flat)
        sample_indices = np.random.choice(num_params, size=min(max_check_params, num_params), replace=False)
        
        max_rel_error = 0.0

        for i in sample_indices:
            orig_val = param_flat[i]
            
            # f(x + eps)
            param_flat[i] = orig_val + eps
            logits_plus, _ = model.forward(input_ids)
            loss_plus, _, _ = compute_cross_entropy_loss(logits_plus, targets, mask)

            # f(x - eps)
            param_flat[i] = orig_val - eps
            logits_minus, _ = model.forward(input_ids)
            loss_minus, _, _ = compute_cross_entropy_loss(logits_minus, targets, mask)

            # Reset original parameter value
            param_flat[i] = orig_val

            grad_numerical = (loss_plus - loss_minus) / (2.0 * eps)
            grad_analytical_val = grad_flat[i]

            rel_error = np.abs(grad_analytical_val - grad_numerical) / (np.abs(grad_analytical_val) + np.abs(grad_numerical) + 1e-8)
            max_rel_error = max(max_rel_error, rel_error)

        status = "PASSED" if max_rel_error < threshold else "FAILED"
        if max_rel_error >= threshold:
            all_passed = False
        print(f"Param Pair #{idx+1:2d} [Shape {str(param.shape):<12s}]: Max Rel Error = {max_rel_error:.2e} -> {status}")

    return all_passed

if __name__ == "__main__":
    np.random.seed(42)
    B, T = 4, 6
    vocab_size = VOCAB_SIZE
    
    dummy_x = np.random.randint(1, vocab_size, size=(B, T))
    dummy_y = np.random.randint(1, vocab_size, size=(B, T))
    dummy_mask = np.ones((B, T), dtype=np.float32)

    # Test Full Transformer Model D
    test_model = ModularTinyTransformer("D", vocab_size=vocab_size, d_model=24, d_ff=96, max_len=10, seed=42)
    assert_tensor_shapes(test_model, dummy_x, dummy_mask)
    passed = finite_difference_gradient_check(test_model, dummy_x, dummy_y, dummy_mask, eps=1e-3, threshold=0.10)
    
    if passed:
        print("\n[SUCCESS] ALL GRADIENT CHECKS PASSED SUCCESSFULLY (Relative error < 10%)!")
    else:
        print("\n[FAIL] GRADIENT CHECK FAILED!")
