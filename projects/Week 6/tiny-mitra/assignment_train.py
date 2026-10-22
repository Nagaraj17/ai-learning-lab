"""Train one instructor-format model locally or in an actual cloud runtime.

The eight-model local study is already registered by assignment_experiments.py.
This command uses a separate output directory/protocol for cloud pilots/runs.
--steps is the TOTAL update count, so 100 then 200 --resume adds 100 updates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from assignment_data import save_dataset
from config import ARCHITECTURES, ModelConfig
from tokenizer import CharacterTokenizer
from train import atomic_json, source_hash, train_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="transformer_2")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0007)
    parser.add_argument("--validate-every", type=int, default=100)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.validate_every) < 1: parser.error("Counts must be positive")
    torch.set_num_threads(2)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no GPU is allocated")
    records, manifest = save_dataset()
    config = ModelConfig(len(CharacterTokenizer()), architecture=args.architecture, context=256)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    own_sources = ("assignment_data.py", "assignment_evaluate.py", "assignment_train.py")
    identity = {"version": "assignment_v2_separate_runtime", "dataset_hash": manifest["dataset_hash"], "core_source_hash": source_hash(), "assignment_source_hash": hashlib.sha256(b"".join(Path(name).read_bytes() for name in own_sources)).hexdigest(), "config": config.to_dict(), "seed": args.seed, "batch_size": args.batch_size, "lr": args.lr, "validate_every": args.validate_every, "device_requested": args.device, "selection": "minimum token-weighted validation loss", "scope": "pilot or separately registered cloud run; not a replacement for original local comparison"}
    protocol = directory / "assignment_protocol.json"
    if protocol.exists() and json.loads(protocol.read_text(encoding="utf-8")) != identity:
        raise ValueError("Saved assignment runtime protocol differs; use a new reviewed directory")
    atomic_json(protocol, identity)
    train_run(config, records, directory, steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed, device=args.device, validate_every=args.validate_every, resume=args.resume)


if __name__ == "__main__":
    main()
