"""Choose an explicitly labelled demo candidate using development evidence only.

The seed42 architecture benchmark stays fixed. This rule was declared after its
10/11 development observation and before the repeat candidates' completed results.
It is exploratory demo preparation, not a claim of unbiased model selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
from train import atomic_json

ROOT = Path("artifacts/mitra_final")
PROTOCOL_PATH = Path("reports/DEMO_SELECTION_PROTOCOL.json")
PROTOCOL = {
    "candidate_seeds_in_priority_order": [42, 7, 123],
    "architecture": "transformer_2",
    "checkpoint": "latest.pt at fixed update4000",
    "acceptance_gate": "all11 authored notes exactly correct with valid EOS plus both additional clinical checks exactly correct with valid EOS",
    "selection": "first passing candidate in declared priority order; otherwise no passing demo candidate",
    "timing_disclosure": "declared after observing seed42 authored10/11 and before either repeat completed; exploratory development selection",
    "benchmark": "primary architecture comparison remains seed42, independent of demo selection",
    "no_final_test_selection": True,
}


def register():
    if PROTOCOL_PATH.exists() and json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")) != PROTOCOL:
        raise ValueError("Demo selection registration differs; preserve and review it")
    atomic_json(PROTOCOL_PATH, PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    register()
    if args.register_only:
        print("Registered development-only demo selection; primary seed42 benchmark unchanged")
        return
    rows = []
    for seed in PROTOCOL["candidate_seeds_in_priority_order"]:
        prefix = "transformer_2" if seed == 42 else f"transformer_2_seed{seed}"
        clinical_name = "clinical_acceptance.json" if seed == 42 else f"clinical_acceptance_seed{seed}.json"
        acceptance = json.loads((ROOT / f"results/{prefix}_acceptance.json").read_text(encoding="utf-8"))
        clinical = json.loads((ROOT / "results" / clinical_name).read_text(encoding="utf-8"))
        assert acceptance["selected_step"] == clinical["checkpoint_step"] == 4000
        assert acceptance["dataset_hash"] == clinical["dataset_hash"] and acceptance["source_hash"] == clinical["source_hash"]
        assert clinical["training_seed"] == seed
        assert len(acceptance["outputs"]) == 11 and len(clinical["outputs"]) == 2
        authored_correct = sum(row["generated"] == row["expected"] and row["protocol_valid"] for row in acceptance["outputs"])
        clinical_correct = sum(row["generated"] == row["expected"] and row["termination"] == "eos" for row in clinical["outputs"])
        path = ROOT / f"checkpoints/transformer_2_seed{seed}/latest.pt"
        rows.append({"seed": seed, "model_id": "task_tokens/transformer_2" if seed == 42 else f"final_seed{seed}/transformer_2", "authored_correct": authored_correct, "authored_notes": 11, "clinical_correct": clinical_correct, "clinical_notes": 2, "passed": authored_correct == 11 and clinical_correct == 2, "checkpoint": path.as_posix(), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    selected = next((row for row in rows if row["passed"]), None)
    result = {"passed": selected is not None, "selected": selected, "candidates": rows, "protocol": PROTOCOL, "scope": "development-only demo selection, not final-test accuracy or production readiness"}
    atomic_json("reports/DEMO_SELECTION.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
