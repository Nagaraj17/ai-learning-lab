"""Prevent silent protocol changes and accidental release of final-test data."""
from types import SimpleNamespace
from pathlib import Path
import pytest
from experiments import freeze_protocol, final_test


def options():
    return SimpleNamespace(series="primary", steps=1000, batch_size=16, lr=0.0003, width=64, seeds=[42], architectures=["embedding_only"], split="validation", limit=12)


def test_registration_creates_fresh_directory_and_rejects_silent_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = options()
    protocol = freeze_protocol([], args)
    assert Path("artifacts/results/protocol.json").exists()
    assert freeze_protocol([], args) == protocol
    args.steps = 2000
    with pytest.raises(ValueError, match="Protocol differs"):
        freeze_protocol([], args)
    args.series = "repeats"
    freeze_protocol([], args)
    assert Path("artifacts/results/protocol_repeats.json").exists()


def test_final_release_requires_review_before_loading_models(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("experiments.load_dataset", lambda: [])
    with pytest.raises(ValueError, match="PROTOCOL_REVIEW"):
        final_test(options())
