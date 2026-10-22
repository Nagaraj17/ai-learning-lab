"""Train one architecture, save best/last states and resume an interrupted run.

Teacher forcing uses expected previous output tokens during training. Gradio
generation feeds the model's own predictions back instead. Never conflate them.
"""
import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from config import ModelConfig, ARCHITECTURES
from data import batch_items, digest, load_dataset
from evaluate import token_metrics
from model import TinyGPT
from tokenizer import CharacterTokenizer


def source_hash():
    files = ("config.py", "data.py", "tokenizer.py", "model.py", "train.py", "generate.py", "evaluate.py")
    return hashlib.sha256(b"".join(name.encode() + (Path(__file__).parent / name).read_bytes() for name in files)).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    replace_with_retry(temporary, path)


def replace_with_retry(temporary, destination):
    """Windows scanners/readers can briefly lock a just-written file.

    Preserve atomic replacement and report failure if the bounded retries fail;
    never silently treat an unsaved checkpoint as durable progress.
    """
    for attempt in range(8):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.025 * 2 ** attempt)


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    replace_with_retry(temporary, path)


def load_checkpoint(path, device="cpu"):
    # Only our project's own trusted dictionary checkpoints should be loaded.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format") != "tiny_mitra_v1":
        raise ValueError("Not a Tiny Mitra checkpoint")
    tokenizer = CharacterTokenizer()
    if checkpoint["tokenizer_hash"] != tokenizer.fingerprint or checkpoint["tokenizer"] != tokenizer.metadata():
        raise ValueError("Tokenizer mismatch")
    config = ModelConfig(**checkpoint["config"])
    if config.vocab_size != len(tokenizer):
        raise ValueError("Vocabulary/config mismatch")
    model = TinyGPT(config).to(device)
    model.load_state_dict(checkpoint["weights"], strict=True)
    model.eval()
    return model, tokenizer, checkpoint


def train_run(config, records, output, steps=1000, batch_size=16, lr=3e-4, seed=42, device="cpu", validate_every=100, resume=False, subset_ids=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    latest, best = output / "latest.pt", output / "best.pt"
    if (latest.exists() or best.exists()) and not resume:
        raise ValueError("Run already exists; use resume or a new run directory")
    torch.manual_seed(seed)
    random.seed(seed)
    sampler = random.Random(seed + 1)
    tokenizer, fingerprint = CharacterTokenizer(), digest(records)
    training = [record for record in records if record["split"] == "train"]
    if subset_ids is not None:
        training = [record for record in training if record["id"] in subset_ids]
    validation = [record for record in records if record["split"] == "validation"]
    if not training or not validation:
        raise ValueError("Training and validation examples required")
    protocol = {"batch_size": batch_size, "lr": lr, "seed": seed, "validate_every": validate_every, "subset_ids": subset_ids}
    model = TinyGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history, start, best_loss, elapsed_before = [], 0, float("inf"), 0.0
    if resume and latest.exists():
        model, tokenizer, saved = load_checkpoint(latest, device)
        if saved["dataset_hash"] != fingerprint or saved["config"] != config.to_dict() or saved["protocol"] != protocol or saved["source_hash"] != source_hash():
            raise ValueError("Resume identity differs: data/config/protocol/source")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        optimizer.load_state_dict(saved["optimizer"])
        sampler.setstate(saved["sampler_state"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if torch.cuda.is_available() and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        start, history, best_loss, elapsed_before = saved["step"], saved["history"], saved["best_validation_loss"], saved["elapsed_seconds"]
    if steps < start:
        raise ValueError("Requested total steps precede saved progress")
    started = time.perf_counter()
    run_id = output.name
    for step in range(start + 1, steps + 1):
        model.train()
        batch = sampler.sample(training, min(batch_size, len(training)))
        x, y, valid = batch_items(batch, tokenizer, config.context, device)
        logits = model(x, valid)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient_norm)):
            atomic_json(output / "failure.json", {"step": step, "loss": float(loss), "gradient_norm": float(gradient_norm)})
            raise FloatingPointError("Nonfinite loss or gradient; inspect failure.json")
        optimizer.step()
        if step % 10 == 0:
            atomic_json(output / "progress.json", {"run_id": run_id, "step": step, "total_steps": steps, "training_loss": float(loss), "elapsed_seconds": elapsed_before + time.perf_counter() - started})
        if step % validate_every == 0 or step == steps:
            validation_metrics = token_metrics(model, validation, tokenizer, batch_size)
            row = {"step": step, "training_loss": float(loss), "gradient_norm": float(gradient_norm), "validation": validation_metrics, "elapsed_seconds": elapsed_before + time.perf_counter() - started}
            history.append(row)
            improved = validation_metrics["loss"] < best_loss
            best_loss = min(best_loss, validation_metrics["loss"])
            checkpoint = {"format": "tiny_mitra_v1", "run_id": run_id, "config": config.to_dict(), "weights": model.state_dict(), "optimizer": optimizer.state_dict(), "tokenizer": tokenizer.metadata(), "tokenizer_hash": tokenizer.fingerprint, "dataset_hash": fingerprint, "source_hash": source_hash(), "protocol": protocol, "step": step, "history": history, "best_validation_loss": best_loss, "selection_rule": "minimum token-weighted validation cross entropy", "sampler_state": sampler.getstate(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "elapsed_seconds": row["elapsed_seconds"], "parameter_count": model.parameter_count, "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": str(device), "gpu": torch.cuda.get_device_name() if str(device).startswith("cuda") else None}}
            save_checkpoint(latest, checkpoint)
            if improved:
                save_checkpoint(best, checkpoint)
            atomic_json(output / "history.json", history)
            print(json.dumps(row), flush=True)
    restored, _, selected = load_checkpoint(best, device)
    # Loading into a new instance must preserve selected validation predictions.
    reloaded_metrics = token_metrics(restored, validation, tokenizer, batch_size)
    atomic_json(output / "manifest.json", {"run_id": run_id, "architecture": config.architecture, "seed": seed, "completed_steps": max(start, steps), "selected_step": selected["step"], "validation": reloaded_metrics, "parameter_count": restored.parameter_count, "dataset_hash": fingerprint, "tokenizer_hash": tokenizer.fingerprint, "source_hash": selected["source_hash"], "protocol": protocol, "config": config.to_dict(), "device": str(device), "status": "trained; generation evaluation separate"})
    return restored, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="transformer_2")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-every", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.steps < 1 or args.batch_size < 1 or args.validate_every < 1:
        parser.error("steps, batch size and validation interval must be positive")
    config = ModelConfig(len(CharacterTokenizer()), architecture=args.architecture, d_model=args.width, context=args.context)
    train_run(config, load_dataset(), args.output or f"artifacts/checkpoints/{args.architecture}_seed{args.seed}", steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed, device=args.device, validate_every=args.validate_every, resume=args.resume)


if __name__ == "__main__":
    main()
