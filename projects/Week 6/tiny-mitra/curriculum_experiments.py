"""Explicit new training stage on broader inputs, starting from our own weights.

This is not an incompatible-data resume: a separately named stage starts at
update0 with parent weights, a fresh optimizer/sampler and recorded lineage.
No outside pretrained model is used. All eight variants follow the same policy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from assignment_data import instructor_examples
from assignment_evaluate import generated_metrics_v2
from broad_data import save_broad_dataset
from config import ARCHITECTURES, ModelConfig
from experiments import diagnostic_subset
from task_runtime import load_task_checkpoint, task_source_hash, train_task
from train import atomic_json, save_checkpoint


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def initialize_stage(parent, directory, records_hash, architecture, producer_hash, parent_steps=2000, parent_data_hash=None):
    lineage_path = directory / "initialization.json"
    parent_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
    lineage = {"kind": "new broader-input stage from our own scratch-trained weights", "parent_checkpoint": str(parent), "parent_checkpoint_sha256": parent_hash, "parent_updates": parent_steps, "parent_dataset_hash": parent_data_hash, "stage_start_update": 0, "optimizer": "fresh AdamW; no parent optimizer state", "sampler": "fresh seed43; no parent data-order state", "stage_dataset_hash": records_hash, "data_producer_source_hash": producer_hash}
    if lineage_path.exists() and read(lineage_path) != lineage: raise ValueError("Recorded stage parent/data differs")
    atomic_json(lineage_path, lineage)
    if (directory / "latest.pt").exists(): return
    if (directory / "best.pt").exists(): raise ValueError("Best exists without latest; preserve and review before recovery")
    torch.manual_seed(42)
    model, tokenizer, metadata = load_task_checkpoint(parent)
    if metadata["step"] != parent_steps or model.config.architecture != architecture: raise ValueError("Parent is not the declared architecture/final update")
    if metadata["source_hash"] != task_source_hash() or parent_data_hash is not None and metadata["dataset_hash"] != parent_data_hash:
        raise ValueError("Parent runtime/data identity differs")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007)
    initial = {"format": "tiny_mitra_task_tokens_v1", "run_id": directory.name, "config": model.config.to_dict(), "weights": model.state_dict(), "optimizer": optimizer.state_dict(), "tokenizer": tokenizer.metadata(), "tokenizer_hash": tokenizer.fingerprint, "dataset_hash": records_hash, "source_hash": task_source_hash(), "protocol": {"seed": 42, "batch_size": 16, "lr": 0.0007, "validate_every": 100}, "step": 0, "history": [], "best_validation_loss": float("inf"), "sampler_state": random.Random(43).getstate(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "elapsed_seconds": 0.0, "parameter_count": model.parameter_count, "initialization": lineage, "environment": {"device": "cpu initialization; selected training device recorded in subsequent checkpoints"}}
    save_checkpoint(directory / "initial_stage.pt", initial)
    save_checkpoint(directory / "latest.pt", initial)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/task_tokens_broad")
    parser.add_argument("--parent-root", default="artifacts/task_tokens")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root, parent_root = Path(args.root), Path(args.parent_root)
    records, data_manifest = save_broad_dataset(root)
    producer_hash = hashlib.sha256(Path("broad_data.py").read_bytes() + Path(__file__).read_bytes()).hexdigest()
    order = ["transformer_2"] + [name for name in ARCHITECTURES if name != "transformer_2"]
    protocol = {"series": "broader-input continued training", "dataset_hash": data_manifest["dataset_hash"], "source_hash": task_source_hash(), "data_producer_source_hash": producer_hash, "architectures": order, "parent_updates": 2000, "stage_updates": 2000, "cumulative_updates": 4000, "batch_size": 16, "lr": 0.0007, "seed": 42, "parent_root": str(parent_root), "selection": "fixed final stage update; minimum-loss weights kept separately", "change": "broader name lengths and identifier digits plus another declared training stage", "limits": "Coverage and added optimization both change; do not attribute the difference solely to one. Authored prompts are development acceptance, not the final partitions."}
    (root / "results").mkdir(parents=True, exist_ok=True)
    path = root / "results/protocol.json"
    if path.exists() and read(path) != protocol: raise ValueError("Continued-stage protocol changed")
    atomic_json(path, protocol)
    source_snapshot = root / "source_snapshot"
    source_snapshot.mkdir(exist_ok=True)
    for name in ("broad_data.py", "curriculum_experiments.py"):
        destination = source_snapshot / name
        content = Path(name).read_bytes()
        if destination.exists() and destination.read_bytes() != content: raise ValueError("Saved producer source differs")
        destination.write_bytes(content)
    validation = diagnostic_subset([row for row in records if row["split"] == "validation"], 24)
    summaries = []
    for architecture in order:
        parent_directory = parent_root / f"checkpoints/{architecture}_seed42"
        while not (parent_directory / "manifest.json").exists() or read(parent_directory / "manifest.json")["completed_steps"] != 2000:
            atomic_json(root / "results/waiting.json", {"status": "waiting for this existing parent run; no duplicate training", "architecture": architecture})
            time.sleep(20)
        directory = root / f"checkpoints/{architecture}_seed42"
        directory.mkdir(parents=True, exist_ok=True)
        initialize_stage(parent_directory / "latest.pt", directory, data_manifest["dataset_hash"], architecture, producer_hash, parent_data_hash=read(parent_root / "dataset/manifest.json")["dataset_hash"])
        config = ModelConfig(data_manifest["vocabulary_size"], architecture=architecture, context=256)
        train_task(config, records, directory, steps=2000, device=args.device, resume=True)
        model, tokenizer, checkpoint = load_task_checkpoint(directory / "latest.pt", args.device)
        reports = {}
        for kind, selected in (("validation", validation), ("acceptance", instructor_examples())):
            report = generated_metrics_v2(model, selected, tokenizer)
            report.update(series="broader-input continued training", architecture=architecture, selected_step=2000, parent_updates=2000, cumulative_updates=4000, dataset_hash=data_manifest["dataset_hash"], source_hash=checkpoint["source_hash"], tokenizer_hash=tokenizer.fingerprint, scope="development evidence; no final-partition tuning")
            atomic_json(root / f"results/{architecture}_{kind}.json", report)
            reports[kind] = report["metrics"]
        last = checkpoint["history"][-1]["validation"]
        summaries.append({"architecture": architecture, "seed": 42, "selected_step": 2000, "updates": 2000, "parent_updates": 2000, "cumulative_updates": 4000, "selection": "fixed final stage update", "parameters": model.parameter_count, "validation_loss": last["loss"], "validation_token_accuracy": last["next_token_accuracy"], "diagnostic_exact": reports["validation"]["exact_match"], "diagnostic_recall": reports["validation"]["recall"], "diagnostic_clinical_preservation": reports["validation"]["clinical_preservation"], "acceptance_correct_notes": reports["acceptance"]["exact"], "acceptance_notes": reports["acceptance"]["examples"]})
        atomic_json(root / "results/summary.json", summaries)
        print(json.dumps(summaries[-1]), flush=True)


if __name__ == "__main__":
    main()
