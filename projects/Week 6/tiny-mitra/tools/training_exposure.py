"""Reconstruct exact scored-token exposure from the saved deterministic sampler.

The saved sampler state must match replay. Equivalent passes are not epochs:
each update samples without replacement inside a batch, with reuse across updates.
"""
import json
import random
from pathlib import Path
import torch
from data import digest, load_dataset, training_item
from tokenizer import CharacterTokenizer
from train import atomic_json, load_checkpoint


def main():
    from assignment_data import create_dataset
    from task_runtime import load_task_checkpoint
    from task_tokenizer import TaskTokenizer
    rows = []
    final = json.loads(Path("artifacts/mitra_final/dataset/examples.json").read_text(encoding="utf-8"))
    studies = (("version_1", Path("artifacts/checkpoints"), load_dataset(), 512, CharacterTokenizer(), load_checkpoint), ("assignment_v2", Path("artifacts/assignment_v2/checkpoints"), create_dataset(), 256, CharacterTokenizer(), load_checkpoint), ("task_tokens", Path("artifacts/task_tokens/checkpoints"), create_dataset(), 256, TaskTokenizer(), load_task_checkpoint), ("mitra_final", Path("artifacts/mitra_final/checkpoints"), final, 256, TaskTokenizer(), load_task_checkpoint))
    for study, root, all_records, context, tokenizer, loader in studies:
        records = [record for record in all_records if record["split"] == "train"]
        lengths = {record["id"]: sum(target != -100 for target in training_item(record, tokenizer, context)[1]) for record in records}
        for directory in sorted(root.glob("*_seed*")):
            if not (directory / "latest.pt").exists(): continue
            _, _, latest = loader(directory / "latest.pt")
            _, _, minimum_loss = loader(directory / "best.pt")
            selected = latest if study.startswith("task_tokens") or study == "mitra_final" else minimum_loss
            if latest["dataset_hash"] != digest(all_records):
                raise ValueError(f"Dataset identity differs for {directory.name}")
            # A stopped extension may have a newer durable checkpoint than its
            # last completed-run manifest. Replay what was actually saved.
            sampler = random.Random(latest["protocol"]["seed"] + 1)
            scored, selected_scored, seen = 0, None, set()
            for step in range(1, latest["step"] + 1):
                batch = sampler.sample(records, min(latest["protocol"]["batch_size"], len(records)))
                scored += sum(lengths[record["id"]] for record in batch)
                seen.update(record["id"] for record in batch)
                if step == selected["step"]: selected_scored = scored
            if sampler.getstate() != latest["sampler_state"]:
                raise ValueError(f"Sampler replay differs for {directory.name}")
            rows.append({"study": study, "run_id": directory.name, "seed": latest["protocol"]["seed"], "durably_saved_updates": latest["step"], "selected_update": selected["step"], "minimum_loss_update": minimum_loss["step"], "scored_training_tokens_completed": scored, "scored_training_tokens_at_selected_update": selected_scored, "example_exposures": latest["step"] * latest["protocol"]["batch_size"], "equivalent_example_passes": latest["step"] * latest["protocol"]["batch_size"] / len(records), "unique_training_examples_seen": len(seen), "sampler_state_replay_matches": True})
    atomic_json("artifacts/results/training_exposure.json", {"procedure": "exact deterministic sampler replay checked against saved state; equivalent passes are not full epochs", "results": rows})
    print(f"Verified deterministic exposure for {len(rows)} runs")


if __name__ == "__main__":
    main()
