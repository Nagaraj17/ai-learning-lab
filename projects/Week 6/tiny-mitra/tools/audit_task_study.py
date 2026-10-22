"""Check the fixed-update task-piece study and report its separate demo gate."""
import json
import argparse
import hashlib
from pathlib import Path
import torch
from assignment_data import create_dataset, instructor_examples
from assignment_evaluate import counts_for_note
from config import ARCHITECTURES
from data import digest
from evaluate import aggregate
from experiments import diagnostic_subset
from task_runtime import load_task_checkpoint, task_source_hash
from train import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/mitra_final")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.root)
    protocol = json.loads((root / "results/protocol.json").read_text(encoding="utf-8"))
    records = json.loads((root / "dataset/examples.json").read_text(encoding="utf-8"))
    assert protocol["source_hash"] == task_source_hash() and protocol["dataset_hash"] == digest(records)
    expected_steps = protocol.get("steps", protocol.get("stage_updates"))
    assert set(protocol["architectures"]) == set(ARCHITECTURES) and expected_steps in (2000, 4000)
    continued = "parent_updates" in protocol
    summary = json.loads((root / "results/summary.json").read_text(encoding="utf-8"))
    assert len(summary) == 8 and {row["architecture"] for row in summary} == set(ARCHITECTURES)
    checks, acceptance_by_model = [], {}
    for architecture in ARCHITECTURES:
        directory = root / f"checkpoints/{architecture}_seed42"
        model, tokenizer, checkpoint = load_task_checkpoint(directory / "latest.pt")
        _, _, minimum_loss = load_task_checkpoint(directory / "best.pt")
        assert checkpoint["step"] == expected_steps and model.config.architecture == architecture
        assert model.config.d_model == protocol.get("width", 64) and model.config.d_ff == protocol.get("ffn", 192) and model.config.context == protocol.get("context", 256)
        assert checkpoint["source_hash"] == task_source_hash() and checkpoint["dataset_hash"] == digest(records)
        if continued:
            lineage = json.loads((directory / "initialization.json").read_text(encoding="utf-8"))
            parent_path = Path(lineage["parent_checkpoint"])
            assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == lineage["parent_checkpoint_sha256"]
            _, _, parent = load_task_checkpoint(parent_path)
            _, _, initial = load_task_checkpoint(directory / "initial_stage.pt")
            assert parent["step"] == 2000 and initial["step"] == 0 and initial["optimizer"]["state"] == {}
            assert initial["dataset_hash"] == digest(records) and initial["dataset_hash"] != parent["dataset_hash"]
            assert all(torch.equal(value, initial["weights"][key]) for key, value in parent["weights"].items())
        row = next(row for row in summary if row["architecture"] == architecture)
        assert row["selected_step"] == expected_steps and row["parameters"] == model.parameter_count
        for kind, candidates in (("validation", [record for record in records if record["split"] == "validation"]), ("acceptance", instructor_examples())):
            if kind == "validation": candidates = diagnostic_subset(candidates, 24)
            report = json.loads((root / f"results/{architecture}_{kind}.json").read_text(encoding="utf-8"))
            assert report["selected_step"] == expected_steps and report["tokenizer_hash"] == tokenizer.fingerprint
            assert report["source_hash"] == checkpoint["source_hash"] and report["dataset_hash"] == digest(records)
            indexed = {record["id"]: record for record in candidates}
            assert sorted(raw["id"] for raw in report["outputs"]) == sorted(indexed), "Duplicate or missing diagnostic note"
            counts = []
            for raw in report["outputs"]:
                record = indexed[raw["id"]]
                assert raw["source"] == record["source"] and raw["expected"] == record["target"]
                measured = counts_for_note(record, raw["generated"])
                assert measured == raw["counts"]
                counts.append(measured)
            measured = aggregate(counts)
            for key, value in measured.items(): assert report["metrics"][key] == value
            assert measured["examples"] == (24 if kind == "validation" else 11)
            assert report["metrics"]["valid_terminations"] == sum(raw["protocol_valid"] for raw in report["outputs"])
            if kind == "acceptance": acceptance_by_model[architecture] = report["metrics"]
        checks.append({"architecture": architecture, "benchmark_checkpoint": "latest.pt", "benchmark_step": expected_steps, "minimum_loss_checkpoint_step": minimum_loss["step"], "parameters": model.parameter_count, "passed": True})
    full = acceptance_by_model["transformer_2"]
    result = {"engineering_audit_passed": True, "demo_ready": full["exact"] == 11 and full["valid_terminations"] == 11, "demo_gate": "full two-block model: all 11 authored development notes exact and valid EOS; held-out generalization is a separate measurement", "full_model_authored_exact": full["exact"], "full_model_authored_notes": 11, "selection_contract": f"Benchmark uses latest.pt at the fixed {expected_steps:,}th update. Trainer manifest.selected_step describes the separately retained minimum-loss best.pt, not the benchmark choice.", "checks": checks, "limits": "Author self-review. Authored acceptance is not final-test performance. Different tokenizers have different per-token units and vocabulary/parameter counts."}
    if protocol.get("producer_files"):
        assert hashlib.sha256(b"".join(Path(name).read_bytes() for name in protocol["producer_files"])).hexdigest() == protocol["data_producer_source_hash"]
        for name in protocol["producer_files"]: assert (root / "source_snapshot" / name).read_bytes() == Path(name).read_bytes()
    if continued:
        producer_hash = hashlib.sha256(Path("broad_data.py").read_bytes() + Path("curriculum_experiments.py").read_bytes()).hexdigest()
        assert producer_hash == protocol["data_producer_source_hash"]
        for name in ("broad_data.py", "curriculum_experiments.py"):
            assert (root / "source_snapshot" / name).read_bytes() == Path(name).read_bytes()
        assert protocol["parent_updates"] == 2000 and protocol["cumulative_updates"] == 4000
        authored = {row["source"] for row in instructor_examples()}
        assert not authored & {row["source"] for row in records if row["split"] == "train"}
        parent_records = json.loads((Path(protocol["parent_root"]) / "dataset/examples.json").read_text(encoding="utf-8"))
        result.update(parent_updates=2000, stage_updates=2000, cumulative_updates=4000, authored_sources_present_in_parent_training=sorted(authored & {row["source"] for row in parent_records if row["split"] == "train"}), authored_sources_present_in_new_stage_training=0, change_limit="Broader inputs and more optimization both change; this is not a tokenizer-only effect")
    atomic_json(root / "results/audit.json", result)
    atomic_json(root / "results/completion.json", {"status": "all eight declared runs completed and engineering audit passed", "demo_quality_gate": result["demo_ready"], "scope": "completion of engineering evidence does not imply broad task generalization or learner mastery"})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
