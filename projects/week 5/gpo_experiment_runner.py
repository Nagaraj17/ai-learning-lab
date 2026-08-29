"""
experiment_runner.py
====================
Training harness for the Week 5 Tiny Transformer experiment.

Features:
- Records full training history (loss, accuracy, grad_norm) per epoch
- Implements early stopping using validation loss
- Gradient clipping
- Macro F1, Top-3 accuracy, per-scenario accuracy computation
- All results come from executed code (never manually typed)
- Saves results as JSON for reproducibility
"""

import numpy as np
import time
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from supply_chain_generator import (
    generate_supply_chain_cases, create_next_token_batches,
    validate_dataset, VOCAB_SIZE, ID2TOKEN, SCENARIO_FAMILIES,
    VAL_ONLY_FAMILIES, TEST_ONLY_FAMILIES
)
from numpy_transformer_suite import (
    ModularTinyTransformer, compute_cross_entropy_loss
)


# ---------------------------------------------------------------------------
# Optimizer utilities
# ---------------------------------------------------------------------------

def compute_grad_norm(param_grad_pairs):
    total_sq = sum(np.sum(g ** 2) for _, g in param_grad_pairs)
    return float(np.sqrt(total_sq))


def clip_gradients(param_grad_pairs, max_norm=1.0):
    norm = compute_grad_norm(param_grad_pairs)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-8)
        for _, g in param_grad_pairs:
            g *= scale
    return norm  # return pre-clip norm


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate_split(model, batches, id2token=None):
    """
    Evaluate model on a set of batches.

    Returns
    -------
    dict with keys:
        loss, top1_acc, top3_acc, macro_f1
        per_class_recall: {token_id: recall}
        all_preds, all_targets
    """
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []
    all_probs_top3 = []

    for batch in batches:
        X, Y, mask = batch["X"], batch["Y"], batch["mask"]
        logits, _ = model.forward(X)
        loss, _, probs = compute_cross_entropy_loss(logits, Y, mask)
        total_loss += loss
        n_batches += 1

        valid_mask = (mask > 0)  # (B, T)
        pred = np.argmax(probs, axis=-1)  # (B, T)

        # Top-3
        top3 = np.argsort(probs, axis=-1)[..., -3:]  # (B, T, 3)

        for b in range(X.shape[0]):
            for t in range(X.shape[1]):
                if valid_mask[b, t]:
                    all_preds.append(int(pred[b, t]))
                    all_targets.append(int(Y[b, t]))
                    all_probs_top3.append(top3[b, t].tolist())

    mean_loss = total_loss / max(n_batches, 1)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Top-1 accuracy
    top1_acc = float(np.mean(all_preds == all_targets)) * 100.0

    # Top-3 accuracy
    top3_correct = sum(
        t in top3 for t, top3 in zip(all_targets.tolist(), all_probs_top3)
    )
    top3_acc = float(top3_correct) / max(len(all_targets), 1) * 100.0

    # Macro F1 and per-class recall
    classes = np.unique(all_targets)
    f1_scores = []
    per_class_recall = {}
    for cls in classes:
        tp = np.sum((all_preds == cls) & (all_targets == cls))
        fp = np.sum((all_preds == cls) & (all_targets != cls))
        fn = np.sum((all_preds != cls) & (all_targets == cls))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        f1_scores.append(f1)
        per_class_recall[int(cls)] = float(recall)

    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    return {
        "loss": float(mean_loss),
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "all_preds": all_preds.tolist(),
        "all_targets": all_targets.tolist(),
    }


def evaluate_per_scenario(model, case_list, max_seq_len=20):
    """
    Evaluate per-scenario accuracy on individual cases.

    Returns dict: {flow_type: {"correct": int, "total": int, "acc": float}}
    """
    scenario_results = {}

    for case in case_list:
        fam = case["flow_type"]
        if fam not in scenario_results:
            scenario_results[fam] = {"correct": 0, "total": 0}

        t_ids = case["token_ids"]
        x_seq = t_ids[:-1]
        y_seq = t_ids[1:]
        if len(x_seq) > max_seq_len - 1:
            x_seq = x_seq[:max_seq_len - 1]
            y_seq = y_seq[:max_seq_len - 1]

        X = np.array([x_seq], dtype=np.int64)
        Y = np.array([y_seq], dtype=np.int64)
        mask = np.ones((1, len(x_seq)), dtype=np.float32)

        logits, _ = model.forward(X)
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)
        preds = np.argmax(probs, axis=-1)

        n_correct = int(np.sum(preds[0] == np.array(y_seq)))
        scenario_results[fam]["correct"] += n_correct
        scenario_results[fam]["total"] += len(y_seq)

    for fam in scenario_results:
        r = scenario_results[fam]
        r["acc"] = r["correct"] / max(r["total"], 1) * 100.0

    return scenario_results


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def train_single_run(
    model_id,
    train_batches, val_batches, test_batches,
    test_cases,
    vocab_size,
    seed,
    lr=0.03,
    max_epochs=800,
    patience=60,
    grad_clip=1.0,
    d_ff=96,
    verbose=False,
):
    """
    Train one model for one seed. Returns full history and final test metrics.

    History recorded per epoch:
        train_loss, val_loss, train_acc, val_acc, grad_norm

    Early stopping: on val_loss (no test-set tuning).
    Best checkpoint restored before final test evaluation.
    """
    model = ModularTinyTransformer(
        model_id, vocab_size=vocab_size,
        d_model=24, d_ff=d_ff, max_len=20, seed=seed
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "grad_norm": []
    }

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0
    best_epoch = 0
    t_start = time.time()

    for epoch in range(max_epochs):
        # --- Training step (batches already shuffled; re-shuffle each epoch) ---
        epoch_train_losses = []
        epoch_train_correct = 0
        epoch_train_total = 0
        epoch_grad_norms = []

        for batch in train_batches:
            X, Y, mask = batch["X"], batch["Y"], batch["mask"]
            logits, x_final = model.forward(X)
            loss, dlogits, probs = compute_cross_entropy_loss(logits, Y, mask)
            epoch_train_losses.append(float(loss))

            preds = np.argmax(probs, axis=-1)
            valid = mask > 0
            epoch_train_correct += int(np.sum((preds == Y) & valid))
            epoch_train_total += int(np.sum(valid))

            model.backward(dlogits, x_final)
            pg = model.get_params_and_grads()
            g_norm = clip_gradients(pg, max_norm=grad_clip)
            epoch_grad_norms.append(g_norm)

            for param, grad in pg:
                param -= lr * grad

        # --- Validation ---
        val_metrics = evaluate_split(model, val_batches)

        t_loss = float(np.mean(epoch_train_losses))
        t_acc = (epoch_train_correct / max(epoch_train_total, 1)) * 100.0
        g_norm_mean = float(np.mean(epoch_grad_norms))

        history["train_loss"].append(t_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(t_acc)
        history["val_acc"].append(val_metrics["top1_acc"])
        history["grad_norm"].append(g_norm_mean)

        if verbose and epoch % 50 == 0:
            print(f"  Epoch {epoch:4d} | train_loss={t_loss:.4f} "
                  f"val_loss={val_metrics['loss']:.4f} "
                  f"val_acc={val_metrics['top1_acc']:.1f}%")

        # --- Early stopping ---
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            best_weights = [p.copy() for p, _ in model.get_params_and_grads()]
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best checkpoint
    if best_weights is not None:
        for idx, (param, _) in enumerate(model.get_params_and_grads()):
            param[:] = best_weights[idx]

    # --- Final test evaluation ---
    test_metrics = evaluate_split(model, test_batches)
    scenario_acc = evaluate_per_scenario(model, test_cases)
    elapsed = time.time() - t_start

    return {
        "model_id": model_id,
        "seed": seed,
        "d_ff": d_ff,
        "stopped_epoch": epoch + 1,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "elapsed_sec": elapsed,
        "n_params": model.count_parameters(),
        "history": history,
        "test_metrics": test_metrics,
        "scenario_acc": scenario_acc,
    }


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------

ARCH_DESCRIPTIONS = {
    "A":        "Embedding + PE + Linear",
    "B":        "A + 1-head Attention",
    "C":        "A + 4-head Attention",
    "D":        "A + 2 Transformer Blocks",
    "D-1":      "A + 1 Transformer Block",
    "D-no-FFN": "2 Blocks, no FFN",
    "D-no-LN":  "2 Blocks, no LayerNorm",
    "D-no-res": "2 Blocks, no Residual",
}


def run_architecture_benchmark(
    splits, all_cases, seeds,
    architectures=None,
    max_seq_len=20, batch_size=32,
    lr=0.03, max_epochs=800, patience=60,
    save_path=None
):
    """
    Run multi-seed benchmark across architectures.
    Uses ONE fixed dataset split, separate model initialization seeds.

    Returns results dict and per-arch summary.
    """
    if architectures is None:
        architectures = ["A", "B", "C", "D", "D-1", "D-no-FFN", "D-no-LN", "D-no-res"]

    # Fixed batches — same split for all architectures
    train_batches = create_next_token_batches(splits["train"], max_seq_len, batch_size, shuffle=True, seed=42)
    val_batches   = create_next_token_batches(splits["val"],   max_seq_len, batch_size, shuffle=False)
    test_batches  = create_next_token_batches(splits["test"],  max_seq_len, batch_size, shuffle=False)
    test_cases    = splits["test"]

    all_results = []
    arch_summary = {}

    for arch in architectures:
        print(f"\n[{arch}] {ARCH_DESCRIPTIONS.get(arch, '')}")
        arch_runs = []

        for seed in seeds:
            print(f"  seed={seed} ...", end=" ", flush=True)
            run = train_single_run(
                arch, train_batches, val_batches, test_batches,
                test_cases, VOCAB_SIZE, seed=seed,
                lr=lr, max_epochs=max_epochs, patience=patience
            )
            all_results.append(run)
            arch_runs.append(run)
            print(f"val={run['best_val_loss']:.4f} "
                  f"test_acc={run['test_metrics']['top1_acc']:.1f}% "
                  f"ep={run['stopped_epoch']}")

        # Aggregate across seeds
        test_losses = [r["test_metrics"]["loss"] for r in arch_runs]
        test_accs   = [r["test_metrics"]["top1_acc"] for r in arch_runs]
        test_f1s    = [r["test_metrics"]["macro_f1"] for r in arch_runs]
        train_losses = [r["history"]["train_loss"][r["best_epoch"]] for r in arch_runs]
        val_losses   = [r["best_val_loss"] for r in arch_runs]

        arch_summary[arch] = {
            "description": ARCH_DESCRIPTIONS.get(arch, arch),
            "n_params": arch_runs[0]["n_params"],
            "mean_test_loss": float(np.mean(test_losses)),
            "std_test_loss":  float(np.std(test_losses)),
            "mean_test_acc":  float(np.mean(test_accs)),
            "std_test_acc":   float(np.std(test_accs)),
            "mean_macro_f1":  float(np.mean(test_f1s)),
            "std_macro_f1":   float(np.std(test_f1s)),
            "mean_train_loss_at_best": float(np.mean(train_losses)),
            "mean_val_loss_at_best":   float(np.mean(val_losses)),
            "gen_gap": float(np.mean(test_losses)) - float(np.mean(train_losses)),
            "mean_stopped_epoch": float(np.mean([r["stopped_epoch"] for r in arch_runs])),
            "mean_elapsed_sec":   float(np.mean([r["elapsed_sec"] for r in arch_runs])),
        }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            # Convert histories (lists) to JSON-serializable form
            json.dump({
                "arch_summary": arch_summary,
                "all_results": [
                    {k: v for k, v in r.items() if k != "history"}
                    for r in all_results
                ]
            }, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return all_results, arch_summary


def run_ffn_width_experiment(
    splits, all_cases, seeds,
    ffn_widths=None,
    max_seq_len=20, batch_size=32,
    lr=0.03, max_epochs=800, patience=60,
    save_path=None
):
    """
    Vary d_ff for model D-1. Fixed architecture, seeds, and dataset split.
    Returns results and summary keyed by d_ff value.
    """
    if ffn_widths is None:
        ffn_widths = [24, 48, 96, 192]

    train_batches = create_next_token_batches(splits["train"], max_seq_len, batch_size, shuffle=True, seed=42)
    val_batches   = create_next_token_batches(splits["val"],   max_seq_len, batch_size, shuffle=False)
    test_batches  = create_next_token_batches(splits["test"],  max_seq_len, batch_size, shuffle=False)
    test_cases    = splits["test"]

    all_results = []
    width_summary = {}

    for d_ff in ffn_widths:
        print(f"\n[FFN width={d_ff} = {d_ff//24}×d_model]")
        width_runs = []

        for seed in seeds:
            print(f"  seed={seed} ...", end=" ", flush=True)
            run = train_single_run(
                "D-1", train_batches, val_batches, test_batches,
                test_cases, VOCAB_SIZE, seed=seed,
                lr=lr, max_epochs=max_epochs, patience=patience,
                d_ff=d_ff
            )
            all_results.append(run)
            width_runs.append(run)
            print(f"val={run['best_val_loss']:.4f} "
                  f"test_acc={run['test_metrics']['top1_acc']:.1f}%")

        test_losses  = [r["test_metrics"]["loss"] for r in width_runs]
        train_losses = [r["history"]["train_loss"][r["best_epoch"]] for r in width_runs]
        test_accs    = [r["test_metrics"]["top1_acc"] for r in width_runs]
        test_f1s     = [r["test_metrics"]["macro_f1"] for r in width_runs]

        width_summary[d_ff] = {
            "d_ff": d_ff,
            "multiplier": d_ff // 24,
            "n_params": width_runs[0]["n_params"],
            "mean_test_loss":  float(np.mean(test_losses)),
            "std_test_loss":   float(np.std(test_losses)),
            "mean_train_loss": float(np.mean(train_losses)),
            "gen_gap": float(np.mean(test_losses)) - float(np.mean(train_losses)),
            "mean_test_acc":   float(np.mean(test_accs)),
            "std_test_acc":    float(np.std(test_accs)),
            "mean_macro_f1":   float(np.mean(test_f1s)),
        }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"ffn_width_summary": width_summary}, f, indent=2)

    return all_results, width_summary


if __name__ == "__main__":
    # Quick smoke test
    all_cases, splits = generate_supply_chain_cases(num_cases=400, seed=42)
    validate_dataset(all_cases, splits)
    results, summary = run_architecture_benchmark(
        splits, all_cases,
        seeds=[42, 7],
        architectures=["A", "D"],
        max_epochs=50,
        patience=10
    )
    for arch, s in summary.items():
        print(f"{arch}: acc={s['mean_test_acc']:.1f}% params={s['n_params']}")
