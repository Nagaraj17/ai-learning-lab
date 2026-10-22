"""Write an actual-device reference, or compare a downloaded checkpoint on CPU.

GPU-to-CPU portability is not passed by a CPU-written reference. Device origins
are retained in the output so a local rehearsal cannot masquerade as cloud proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from generate import generate
from train import atomic_json, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-reference", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/transformer_2_seed42/best.pt")
    parser.add_argument("--reference", default="artifacts/results/portability_reference.json")
    parser.add_argument("--tokenizer-kind", choices=("character", "task"), default="character")
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = Path(args.checkpoint)
    loader = load_checkpoint
    if args.tokenizer_kind == "task":
        from task_runtime import load_task_checkpoint
        loader = load_task_checkpoint
    model, tokenizer, metadata = loader(checkpoint, args.device)
    source = "PATIENT John Lee DIAGNOSIS NSCLC BIOMARKER EGFR TREATMENT DrugA"
    ids = tokenizer.encode(tokenizer.prompt(source))
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=args.device))[0, -1].cpu()
    result = generate(model, tokenizer, source)
    current = {"device": str(next(model.parameters()).device), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "tokenizer_hash": tokenizer.fingerprint, "config": metadata["config"], "source": source, "ids": ids, "logits": logits.tolist(), "greedy_text": result["text"], "termination": result["termination"]}
    if args.write_reference:
        Path(args.reference).parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.reference, current)
        print(f"Wrote actual {current['device']} reference; CPU verification is a separate invocation")
        return
    previous = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    for key in ("checkpoint_sha256", "tokenizer_hash", "config", "source", "ids"):
        if previous[key] != current[key]: raise ValueError(f"Portability identity differs: {key}")
    reference_logits = torch.tensor(previous["logits"])
    close = bool(torch.allclose(logits, reference_logits, atol=1e-4, rtol=1e-4))
    same = previous["greedy_text"] == current["greedy_text"] and previous["termination"] == current["termination"]
    report = {"reference_device": previous["device"], "current_device": current["device"], "logits_close": close, "maximum_logit_difference": float((logits - reference_logits).abs().max()), "same_greedy_continuation": same, "gpu_to_cpu_passed": previous["device"].startswith("cuda") and current["device"] == "cpu" and close and same, "scope": "one recorded synthetic probe; repeat/expand if decisions differ"}
    atomic_json("artifacts/results/portability_check.json", report)
    print(json.dumps(report, indent=2))
    if not close or not same: raise SystemExit("Portability differs; inspect margins and generated decisions")


if __name__ == "__main__":
    main()
