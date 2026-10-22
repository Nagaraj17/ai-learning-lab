"""Small source-dependent overfit test, separate from the 1,000-example benchmark."""
import json
import argparse
import time
from pathlib import Path
import torch
from config import ModelConfig
from data import render
from generate import generate
from tokenizer import CharacterTokenizer
from train import atomic_json, train_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="artifacts/checkpoints/pilot_overfit")
    parser.add_argument("--report", default="artifacts/results/pilot.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available(): raise RuntimeError("CUDA requested but no GPU allocated")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    Path("artifacts/results").mkdir(parents=True, exist_ok=True)
    records = []
    for index, (name, drug) in enumerate((("Jo", "DrugA"), ("Li", "DrugB"), ("Mo", "DrugA"), ("Bo", "DrugB"), ("Xi", "DrugA"), ("Yu", "DrugB"))):
        record = render([("PATIENT", name), ("TREATMENT", drug)])
        record.update(id=f"pilot_{index}", split="train" if index < 4 else "validation", no_identifiers=False)
        records.append(record)
    output = Path(args.output)
    tokenizer = CharacterTokenizer()
    config = ModelConfig(len(tokenizer), d_model=32, d_ff=96, context=128)
    started = time.perf_counter()
    model, tokenizer = train_run(config, records, output, steps=500, batch_size=4, lr=0.003, validate_every=50, resume=(output / "latest.pt").exists(), device=args.device)
    rows = []
    for record in records:
        result = generate(model, tokenizer, record["source"], max_new_tokens=80)
        rows.append({"split": record["split"], "source": record["source"], "expected": record["target"], "generated": result["text"], "termination": result["termination"], "exact": result["text"] == record["target"]})
    passed = all(row["exact"] and row["termination"] == "eos" for row in rows[:4])
    if args.device.startswith("cuda"): torch.cuda.synchronize()
    report = {"purpose": "source-dependent tiny-batch learning, not final-test performance", "passed": passed, "elapsed_seconds_this_invocation": time.perf_counter() - started, "outputs": rows, "device": str(next(model.parameters()).device), "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None, "budget": {"steps": 500, "width": 32, "batch": 4, "lr": 0.003}}
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit("Tiny-batch free generation has not passed; investigate before benchmark")


if __name__ == "__main__":
    main()
