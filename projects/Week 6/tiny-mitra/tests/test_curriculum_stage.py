"""A new-data stage must preserve weights while resetting optimizer/progress."""
import json
import pytest
import torch
from config import ModelConfig
from curriculum_experiments import initialize_stage
from data import digest, render
from task_runtime import load_task_checkpoint, train_task
from task_tokenizer import TaskTokenizer


def test_new_stage_is_explicit_weight_initialization_not_incompatible_resume(tmp_path):
    torch.set_num_threads(2)
    records = []
    for index, name in enumerate(("Jo", "Li", "Xi")):
        row = render([("PATIENT", name), ("TREATMENT", "DrugA")])
        row.update(id=str(index), split="validation" if index == 2 else "train")
        records.append(row)
    config = ModelConfig(len(TaskTokenizer()), d_model=16, d_ff=32, context=128)
    parent = tmp_path / "parent"
    train_task(config, records, parent, steps=2, validate_every=2)
    _, _, before = load_task_checkpoint(parent / "latest.pt")
    changed = [dict(row, **render([("PATIENT", name), ("TREATMENT", "DrugA")])) for row, name in zip(records, ("Jordan", "Li", "Xi"))]
    stage = tmp_path / "new_stage"
    stage.mkdir()
    initialize_stage(parent / "latest.pt", stage, digest(changed), "transformer_2", "fixture_producer", parent_steps=2, parent_data_hash=digest(records))
    _, _, initial = load_task_checkpoint(stage / "latest.pt")
    assert initial["step"] == 0 and initial["history"] == [] and initial["optimizer"]["state"] == {}
    assert before["optimizer"]["state"] and initial["dataset_hash"] != before["dataset_hash"]
    assert all(torch.equal(value, initial["weights"][name]) for name, value in before["weights"].items())
    assert initial["protocol"]["validate_every"] == 100
    train_task(config, changed, stage, steps=2, resume=True)
    _, _, after = load_task_checkpoint(stage / "latest.pt")
    assert after["step"] == 2 and len(after["history"]) == 1 and after["optimizer"]["state"]
    assert any(not torch.equal(value, after["weights"][name]) for name, value in initial["weights"].items())
    lineage = json.loads((stage / "initialization.json").read_text())
    assert lineage["parent_updates"] == 2 and lineage["stage_start_update"] == 0
    with pytest.raises(ValueError, match="Recorded stage parent/data differs"):
        initialize_stage(parent / "latest.pt", stage, "different-data", "transformer_2", "fixture_producer", parent_steps=2)
