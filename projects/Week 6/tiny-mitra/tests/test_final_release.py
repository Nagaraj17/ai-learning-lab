"""The new-tokenizer final test must retain its own explicit release gate."""
import json
import pytest
from tools.final_partitions import run_final_partitions
from data import digest
from task_runtime import task_source_hash


def test_final_release_requires_review_before_touching_output(tmp_path):
    with pytest.raises(ValueError, match="saved passed review"):
        run_final_partitions(tmp_path, tmp_path / "missing.json")
    assert not (tmp_path / "results/final_test").exists()


def test_changed_final_review_is_rejected_before_loading_any_weights(tmp_path):
    (tmp_path / "dataset").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "dataset/examples.json").write_text("[]")
    (tmp_path / "results/protocol.json").write_text(json.dumps({"data_producer_source_hash": "fixture"}))
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"status": "passed", "no_test_tuning": True, "source_hash": "different-source"}))
    with pytest.raises(ValueError, match="identities differ"):
        run_final_partitions(tmp_path, review)
    assert not (tmp_path / "results/final_test").exists()


def test_protocol_review_alone_cannot_release_unaudited_models(tmp_path):
    (tmp_path / "dataset").mkdir()
    (tmp_path / "results").mkdir()
    protocol = {"data_producer_source_hash": "fixture", "source_hash": task_source_hash(), "dataset_hash": digest([])}
    (tmp_path / "dataset/examples.json").write_text("[]")
    (tmp_path / "results/protocol.json").write_text(json.dumps(protocol))
    review = tmp_path / "review.json"
    review.write_text(json.dumps(dict(protocol, protocol_hash=digest(protocol), status="passed", no_test_tuning=True)))
    with pytest.raises(ValueError, match="Completed-model engineering audit"):
        run_final_partitions(tmp_path, review)
    assert not (tmp_path / "results/final_test").exists()
