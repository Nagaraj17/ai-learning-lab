"""One fixed-task-piece run, including a cloud pilot that can resume100→200."""
import argparse
import json
from pathlib import Path
import torch
from assignment_data import create_dataset
from config import ARCHITECTURES, ModelConfig
from task_runtime import train_task
from task_tokenizer import TaskTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="transformer_2")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--dataset", type=Path, help="A saved synthetic dataset; default is the initial instructor-format corpus")
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = ModelConfig(len(TaskTokenizer()), architecture=args.architecture, context=256)
    records = json.loads(args.dataset.read_text(encoding="utf-8")) if args.dataset else create_dataset()
    train_task(config, records, args.output, steps=args.steps, device=args.device, validate_every=args.validate_every, resume=args.resume)


if __name__ == "__main__":
    main()
