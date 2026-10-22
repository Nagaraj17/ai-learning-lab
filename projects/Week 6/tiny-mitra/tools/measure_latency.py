"""Measure warmed CPU inference after the training/evaluation workers finish."""
import json
import statistics
import time
from pathlib import Path
import torch
from config import ARCHITECTURES
from generate import generate
from train import atomic_json, load_checkpoint


def main():
    torch.set_num_threads(2)
    source = "PATIENT John Lee DIAGNOSIS NSCLC BIOMARKER EGFR TREATMENT DrugA"
    rows = []
    choices = [("version_1", name, Path(f"artifacts/checkpoints/{name}_seed42/best.pt")) for name in ARCHITECTURES]
    choices += [("assignment_v2", name, Path(f"artifacts/assignment_v2/checkpoints/{name}_seed42/best.pt")) for name in ARCHITECTURES]
    choices += [("task_tokens", name, Path(f"artifacts/task_tokens/checkpoints/{name}_seed42/latest.pt")) for name in ARCHITECTURES]
    choices += [("mitra_final", name, Path(f"artifacts/mitra_final/checkpoints/{name}_seed42/latest.pt")) for name in ARCHITECTURES]
    choices += [("mitra_final", f"transformer_2_seed{seed}", Path(f"artifacts/mitra_final/checkpoints/transformer_2_seed{seed}/latest.pt")) for seed in (7, 123)]
    choices += [("four_note_pilot", "transformer_2", Path("artifacts/checkpoints/pilot_overfit/best.pt"))]
    for study, architecture, path in choices:
        started = time.perf_counter()
        loader = load_checkpoint
        if study.startswith("task_tokens") or study == "mitra_final":
            from task_runtime import load_task_checkpoint
            loader = load_task_checkpoint
        model, tokenizer, checkpoint = loader(path)
        load_seconds = time.perf_counter() - started
        generate(model, tokenizer, source, max_new_tokens=1)
        samples = []
        for _ in range(3):
            started = time.perf_counter()
            result = generate(model, tokenizer, source)
            elapsed = time.perf_counter() - started
            samples.append({"seconds": elapsed, "tokens": len(result["tokens"]), "termination": result["termination"]})
        median = statistics.median(row["seconds"] for row in samples)
        tokens = samples[0]["tokens"]
        rows.append({"study": study, "architecture": architecture, "training_seed": checkpoint["protocol"]["seed"], "step": checkpoint["step"], "load_seconds": load_seconds, "median_seconds_per_note": median, "generated_tokens": tokens, "tokens_per_second": tokens / median, "samples": samples})
    atomic_json("artifacts/results/quiet_cpu_latency.json", {"device": "cpu", "torch_threads": 2, "torch_version": str(torch.__version__), "source": source, "procedure": "after queued training/evaluation processes finished; warm once; three greedy repetitions; load time separate; generated length varies by checkpoint", "results": rows})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
