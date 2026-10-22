"""A transient Windows lock must not discard a checkpoint or progress update."""
from pathlib import Path
from train import atomic_json


def test_atomic_progress_retries_transient_file_lock(tmp_path, monkeypatch):
    original = Path.replace
    calls = []
    def locked_twice(self, target):
        calls.append(str(target))
        if len(calls) < 3:
            raise PermissionError("simulated brief reader/scanner lock")
        return original(self, target)
    monkeypatch.setattr(Path, "replace", locked_twice)
    destination = tmp_path / "progress.json"
    atomic_json(destination, {"step": 10})
    assert len(calls) == 3
    assert destination.read_text().strip() == '{\n  "step": 10\n}'
