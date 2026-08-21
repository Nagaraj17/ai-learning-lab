import numpy as np
import time
from step_therapy_generator import generate_step_therapy_cases, create_dataset_batches, VOCAB_SIZE
from numpy_transformer_suite import ModularTinyTransformer, compute_cross_entropy_loss

def clip_gradients(param_grad_pairs, max_norm=1.0):
    total_norm_sq = 0.0
    for param, grad in param_grad_pairs:
        total_norm_sq += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm_sq)
    
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        for param, grad in param_grad_pairs:
            grad *= scale

def train_single_run(model_id, train_batches, val_batches, test_batches, vocab_size, seed,
                     lr=0.03, max_epochs=800, patience=60, grad_clip=1.0):
    model = ModularTinyTransformer(model_id, vocab_size=vocab_size, d_model=24, d_ff=96, max_len=10, seed=seed)
    
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # Training loop
        train_losses = []
        for batch in train_batches:
            X, Y, mask = batch["X"], batch["Y"], batch["mask"]
            logits, x_final = model.forward(X)
            loss, dlogits, _ = compute_cross_entropy_loss(logits, Y, mask)
            train_losses.append(loss)
            
            model.backward(dlogits, x_final)
            param_grad_pairs = model.get_params_and_grads()
            
            # Gradient clipping
            clip_gradients(param_grad_pairs, max_norm=grad_clip)
            
            # SGD update
            for param, grad in param_grad_pairs:
                param -= lr * grad
                
        avg_train_loss = np.mean(train_losses)
        
        # Validation check
        val_losses = []
        for batch in val_batches:
            X, Y, mask = batch["X"], batch["Y"], batch["mask"]
            logits, _ = model.forward(X)
            loss, _, _ = compute_cross_entropy_loss(logits, Y, mask)
            val_losses.append(loss)
            
        avg_val_loss = np.mean(val_losses)
        
        # Early Stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best parameters snapshot
            best_weights = [(param.copy()) for param, _ in model.get_params_and_grads()]
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best weights for final test evaluation
    param_grad_pairs = model.get_params_and_grads()
    for idx, (param, _) in enumerate(param_grad_pairs):
        param[:] = best_weights[idx]
        
    # Evaluate on held-out Test Set
    test_losses = []
    correct_tokens = 0
    total_tokens = 0
    
    for batch in test_batches:
        X, Y, mask = batch["X"], batch["Y"], batch["mask"]
        logits, _ = model.forward(X)
        loss, _, probs = compute_cross_entropy_loss(logits, Y, mask)
        test_losses.append(loss)
        
        preds = np.argmax(probs, axis=-1)
        valid_mask = (mask > 0)
        correct_tokens += np.sum((preds == Y) & valid_mask)
        total_tokens += np.sum(valid_mask)
        
    final_test_loss = np.mean(test_losses)
    final_test_acc = (correct_tokens / total_tokens) * 100.0 if total_tokens > 0 else 0.0

    return {
        "model_id": model_id,
        "seed": seed,
        "stopped_epoch": epoch + 1,
        "best_val_loss": best_val_loss,
        "test_loss": final_test_loss,
        "test_acc": final_test_acc
    }

def run_full_benchmark(num_cases=1000, seeds=[7, 19, 42, 73, 101]):
    model_matrix = ["A", "B", "C", "D", "D-1", "D-no-FFN", "D-no-LN"]
    print("=" * 80)
    print(f"[START] WEEK 5 TINY TRANSFORMER BENCHMARK ({num_cases} Cases, 5 Seeds)")
    print("=" * 80)

    summary_results = {}

    for model_id in model_matrix:
        print(f"\n--- Benchmarking Model Variant {model_id} ---")
        run_metrics = []
        start_time = time.time()
        
        for seed in seeds:
            # Generate dataset splits with fixed seed schedule
            cases, splits = generate_step_therapy_cases(num_cases=num_cases, seed=seed)
            train_batches = create_dataset_batches(splits["train"], batch_size=32, shuffle=True, seed=seed)
            val_batches = create_dataset_batches(splits["val"], batch_size=32, shuffle=False)
            test_batches = create_dataset_batches(splits["test"], batch_size=32, shuffle=False)
            
            res = train_single_run(model_id, train_batches, val_batches, test_batches, VOCAB_SIZE, seed=seed)
            run_metrics.append(res)
            print(f"  Seed {seed:3d} | Epochs: {res['stopped_epoch']:3d} | Val Loss: {res['best_val_loss']:.4f} | Test Loss: {res['test_loss']:.4f} | Test Acc: {res['test_acc']:.2f}%")

        elapsed = time.time() - start_time
        
        test_losses = [r["test_loss"] for r in run_metrics]
        test_accs = [r["test_acc"] for r in run_metrics]
        
        summary_results[model_id] = {
            "mean_test_loss": np.mean(test_losses),
            "std_test_loss": np.std(test_losses),
            "mean_test_acc": np.mean(test_accs),
            "std_test_acc": np.std(test_accs),
            "elapsed_sec": elapsed
        }

    # Print Final Benchmark Summary Table
    print("\n" + "=" * 90)
    print("FINAL WEEK 5 GENERALIZATION BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'Model ID':<10} | {'Architecture Description':<40} | {'Test Loss (Mean +/- Std)':<25} | {'Test Accuracy (%)':<18}")
    print("-" * 90)

    descriptions = {
        "A": "Embedding + Positional + Linear Head",
        "B": "A + 1 Causal Attention Head",
        "C": "A + 4 Causal Attention Heads",
        "D": "A + 2 Pre-LN Transformer Blocks",
        "D-1": "1 Transformer Block",
        "D-no-FFN": "2 Blocks without FFN",
        "D-no-LN": "2 Blocks without LayerNorm"
    }

    for model_id in model_matrix:
        stats = summary_results[model_id]
        desc = descriptions[model_id]
        loss_str = f"{stats['mean_test_loss']:.4f} +/- {stats['std_test_loss']:.4f}"
        acc_str = f"{stats['mean_test_acc']:.2f}% +/- {stats['std_test_acc']:.2f}%"
        print(f"{model_id:<10} | {desc:<40} | {loss_str:<25} | {acc_str:<18}")

    print("=" * 90)
    return summary_results

if __name__ == "__main__":
    results = run_full_benchmark(num_cases=1000, seeds=[7, 19, 42, 73, 101])
