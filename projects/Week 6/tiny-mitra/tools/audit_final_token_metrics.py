"""Recheck saved teacher-forced test metrics; never train or choose weights."""
import hashlib
import json
from pathlib import Path
import torch
from config import ARCHITECTURES
from data import digest
from evaluate import token_metrics
from task_runtime import load_task_checkpoint
from train import atomic_json


def main():
    torch.set_num_threads(2)
    root = Path("artifacts/mitra_final")
    assert json.loads((root / "results/final_test/audit.json").read_text())["passed"]
    records = json.loads((root / "dataset/examples.json").read_text())
    rows = []
    for name in ARCHITECTURES:
        path = root / f"checkpoints/{name}_seed42/latest.pt"
        model, tokenizer, checkpoint = load_task_checkpoint(path)
        assert checkpoint["step"] == 4000 and checkpoint["dataset_hash"] == digest(records)
        for split in sorted({r["split"] for r in records if r["split"].startswith("test_")}):
            raw = json.loads((root / f"results/final_test/{name}_{split}.json").read_text())
            assert raw["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            measured = token_metrics(model, [r for r in records if r["split"] == split], tokenizer)
            assert measured.keys() == raw["teacher_forced"].keys()
            for key, value in measured.items():
                assert abs(value - raw["teacher_forced"][key]) < 1e-7, (name, split, key)
            rows.append(dict(architecture=name, split=split, checkpoint_sha256=raw["checkpoint_sha256"], **measured))
    atomic_json(root / "results/final_test/token_metrics_audit.json", {"passed": True, "rows": rows, "scope": "32 recomputed output/EOS-token-weighted teacher-forced metrics; correct prefixes are supplied, unlike free generation; no fitting or checkpoint selection"})
    print("Recomputed and matched all32 saved test loss/next-token accuracy rows")


if __name__ == "__main__":
    main()
