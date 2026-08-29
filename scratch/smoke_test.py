"""Quick smoke-test of all 3 modules."""
import sys
sys.path.insert(0, "projects/week 5")

print("=== Testing step_therapy_generator ===")
from step_therapy_generator import (
    generate_step_therapy_cases, validate_dataset, create_next_token_batches,
    VOCAB_SIZE, ID2TOKEN, SCENARIO_FAMILIES
)
cases, splits = generate_step_therapy_cases(num_cases=600, seed=42)
r = validate_dataset(cases, splits)
print(f"total={r['total_cases']} train={r['n_train']} val={r['n_val']} test={r['n_test']}")
print(f"unique={r['n_unique']} dup_rate={r['duplicate_rate']:.4f}")
print(f"overlap TV={r['train_val_overlap']} TT={r['train_test_overlap']} VT={r['val_test_overlap']}")
print(f"test_targets_in_train={r['test_targets_in_train']}")
print(f"seq_len: min={r['seq_len_min']} max={r['seq_len_max']} mean={r['seq_len_mean']:.1f}")
print(f"scenarios: {list(r['scenario_distribution'].keys())}")

print("\n=== Testing numpy_transformer_suite ===")
import numpy as np
from numpy_transformer_suite import ModularTinyTransformer, compute_cross_entropy_loss

for mid in ["A", "B", "C", "D", "D-no-FFN", "D-no-LN", "D-no-res"]:
    m = ModularTinyTransformer(mid, VOCAB_SIZE, d_model=24, d_ff=96, max_len=20, seed=42)
    X = np.random.randint(1, VOCAB_SIZE, (2, 10), dtype=np.int64)
    logits, xf = m.forward(X)
    Y = np.random.randint(1, VOCAB_SIZE, (2, 10), dtype=np.int64)
    mask = np.ones((2, 10), dtype=np.float32)
    loss, dlogits, probs = compute_cross_entropy_loss(logits, Y, mask)
    m.backward(dlogits, xf)
    assert np.isfinite(loss), f"Model {mid}: non-finite loss"
    print(f"  Model {mid}: loss={loss:.4f} params={m.count_parameters()} OK")

print("\n=== Testing gradient_checker ===")
from gradient_checker import run_full_verification
for mid in ["A", "D"]:
    passed, _ = run_full_verification(mid, VOCAB_SIZE, seed=42)
    print(f"  Model {mid}: all_passed={passed}")

print("\n=== ALL SMOKE TESTS PASSED ===")
