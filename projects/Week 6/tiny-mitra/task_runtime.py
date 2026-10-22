"""Explicit training/checkpoint adapter for the task-word tokenizer experiment.

The manual architecture, batching, teacher-forced metrics and generation are
shared with the character study. Its original trainer/loader stays unchanged
while those registered runs finish. The update loop is written here explicitly
so a different tokenizer is never installed through hidden monkey-patching.
"""
import hashlib
import json
from pathlib import Path
import platform
import random
import time
import torch
import torch.nn.functional as F
from config import ModelConfig
from data import batch_items, digest
from evaluate import token_metrics
from model import TinyGPT
from task_tokenizer import TaskTokenizer
from train import atomic_json, save_checkpoint, source_hash


def task_source_hash():
    additional = ("assignment_data.py", "assignment_evaluate.py", "task_tokenizer.py", "task_runtime.py", "task_experiments.py")
    return hashlib.sha256(source_hash().encode() + b"".join(name.encode() + (Path(__file__).parent / name).read_bytes() for name in additional)).hexdigest()


def load_task_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    tokenizer = TaskTokenizer()
    if checkpoint.get("format") != "tiny_mitra_task_tokens_v1" or checkpoint["tokenizer"] != tokenizer.metadata() or checkpoint["tokenizer_hash"] != tokenizer.fingerprint:
        raise ValueError("Task-token checkpoint/tokenizer identity differs")
    config = ModelConfig(**checkpoint["config"])
    if config.vocab_size != len(tokenizer): raise ValueError("Vocabulary/config mismatch")
    model = TinyGPT(config).to(device)
    model.load_state_dict(checkpoint["weights"], strict=True)
    model.eval()
    return model, tokenizer, checkpoint


def train_task(config, records, directory, steps=2000, seed=42, batch_size=16, lr=0.0007, validate_every=100, device="cpu", resume=False):
    if min(steps, batch_size, validate_every) < 1: raise ValueError("Training counts must be positive")
    if str(device).startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but no GPU allocated")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    best, latest = directory / "best.pt", directory / "latest.pt"
    if (best.exists() or latest.exists()) and not resume: raise ValueError("Existing run requires explicit resume")
    torch.manual_seed(seed)
    sampler = random.Random(seed + 1)
    tokenizer = TaskTokenizer()
    training = [row for row in records if row["split"] == "train"]
    validation = [row for row in records if row["split"] == "validation"]
    if not training or not validation: raise ValueError("Both train and validation records required")
    protocol = {"seed": seed, "batch_size": batch_size, "lr": lr, "validate_every": validate_every}
    model = TinyGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    start, history, best_loss, elapsed_before = 0, [], float("inf"), 0.0
    if resume and latest.exists():
        model, tokenizer, saved = load_task_checkpoint(latest, device)
        if saved["dataset_hash"] != digest(records) or saved["source_hash"] != task_source_hash() or saved["config"] != config.to_dict() or saved["protocol"] != protocol:
            raise ValueError("Resume data/source/config/protocol differs")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        optimizer.load_state_dict(saved["optimizer"])
        sampler.setstate(saved["sampler_state"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if torch.cuda.is_available() and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        start, history, best_loss, elapsed_before = saved["step"], saved["history"], saved["best_validation_loss"], saved["elapsed_seconds"]
    if steps < start: raise ValueError("Requested total updates precede saved progress")
    started = time.perf_counter()
    for step in range(start + 1, steps + 1):
        model.train()
        batch = sampler.sample(training, min(batch_size, len(training)))
        inputs, targets, valid = batch_items(batch, tokenizer, config.context, device)
        logits = model(inputs, valid)
        loss = F.cross_entropy(logits.reshape(-1, len(tokenizer)), targets.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient_norm)):
            atomic_json(directory / "failure.json", {"step": step, "loss": float(loss), "gradient_norm": float(gradient_norm)})
            raise FloatingPointError("Nonfinite training value")
        optimizer.step()
        if step % 10 == 0:
            atomic_json(directory / "progress.json", {"step": step, "total_steps": steps, "training_loss": float(loss)})
        if step % validate_every == 0 or step == steps:
            validation_metrics = token_metrics(model, validation, tokenizer, batch_size)
            row = {"step": step, "training_loss": float(loss), "gradient_norm": float(gradient_norm), "validation": validation_metrics, "elapsed_seconds": elapsed_before + time.perf_counter() - started}
            history.append(row)
            improved = validation_metrics["loss"] < best_loss
            best_loss = min(best_loss, validation_metrics["loss"])
            saved = {"format": "tiny_mitra_task_tokens_v1", "run_id": directory.name, "config": config.to_dict(), "weights": model.state_dict(), "optimizer": optimizer.state_dict(), "tokenizer": tokenizer.metadata(), "tokenizer_hash": tokenizer.fingerprint, "dataset_hash": digest(records), "source_hash": task_source_hash(), "protocol": protocol, "step": step, "history": history, "best_validation_loss": best_loss, "sampler_state": sampler.getstate(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "elapsed_seconds": row["elapsed_seconds"], "parameter_count": model.parameter_count, "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": str(device), "gpu": torch.cuda.get_device_name() if str(device).startswith("cuda") else None}}
            save_checkpoint(latest, saved)
            if improved: save_checkpoint(best, saved)
            atomic_json(directory / "history.json", history)
            print(json.dumps(row), flush=True)
    restored, tokenizer, selected = load_task_checkpoint(best, device)
    validation_metrics = token_metrics(restored, validation, tokenizer, batch_size)
    atomic_json(directory / "manifest.json", {"architecture": config.architecture, "seed": seed, "completed_steps": steps, "selected_step": selected["step"], "parameter_count": restored.parameter_count, "validation": validation_metrics, "dataset_hash": digest(records), "source_hash": selected["source_hash"], "tokenizer_hash": tokenizer.fingerprint, "config": config.to_dict(), "protocol": protocol, "device": str(device)})
    return restored, tokenizer, selected
