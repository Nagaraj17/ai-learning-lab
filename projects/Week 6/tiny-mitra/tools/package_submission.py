"""Build one clean, runnable final submission using an explicit file list.

Nothing is trained, deleted, renamed or changed in the working studies. Python
sources are copied byte-for-byte because the registered runs hash their bytes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import quote
import zipfile

from config import ARCHITECTURES
from task_runtime import task_source_hash

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
FROZEN_SOURCE = "bf6418a89068cb5d0d457573435518a39a23d17c7d074d2c87b388533cb3b173"
CORE = (
    "app.py", "assignment_data.py", "assignment_evaluate.py", "broad_data.py",
    "config.py", "curriculum_experiments.py", "data.py", "diagnostics.py",
    "evaluate.py", "experiments.py", "generate.py", "live_generation.py",
    "model.py", "submission_experiments.py", "submission_repeats.py",
    "task_experiments.py", "task_runtime.py", "task_tokenizer.py", "task_train.py",
    "tokenizer.py", "train.py", ".gitignore", ".gitattributes",
    "requirements.txt", "requirements-cpu-tested.txt",
)
TOOLS = (
    "pilot.py", "final_partitions.py", "investigate_task.py",
    "context_architectures.py", "audit_final_partitions.py",
    "audit_final_repeats.py", "audit_final_token_metrics.py",
    "audit_task_investigations.py", "select_demo_checkpoint.py",
    "clinical_acceptance.py", "evaluate_selected_demo.py",
    "review_final_findings.py", "render_final_comparison.py", "verify_submission.py",
)
REPORTS = (
    "FINAL_FINDINGS.md", "FINAL_PROTOCOL_REVIEW.json", "DEMO_SCRIPT.md",
    "DEMO_SELECTION.json", "DEMO_SELECTION_PROTOCOL.json",
    "final_hypothesis_reviews.json", "final_architecture_quality.png",
    "final_head_effects.png", "final_training_curves.png",
)


def package(output=None, archive=None):
    if task_source_hash() != FROZEN_SOURCE:
        raise ValueError("Frozen model/training source changed; review a new experiment first")
    output = Path(output) if output else ROOT / "artifacts/submission/tiny-mitra-final-submission"
    archive = Path(archive) if archive else ROOT / "artifacts/bundles/tiny-mitra-final-submission.zip"
    output, archive = output.resolve(), archive.resolve()
    if output.exists() or archive.exists():
        raise FileExistsError("Preserve the existing package; choose new --output and --archive paths")
    # Keep package destinations out of source trees and out of one another.
    allowed = ROOT / "artifacts"
    if not output.is_relative_to(allowed) or not archive.is_relative_to(allowed):
        raise ValueError("Package destinations must stay within this project's artifacts directory")
    if archive.is_relative_to(output):
        raise ValueError("ZIP cannot be inside the folder it archives")
    selected = {}

    def add(source, target, reason):
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        if target in selected:
            raise ValueError("Duplicate destination: " + target)
        selected[target] = (source, reason)

    add(ROOT / "submission/README.md", "README.md", "single submission entry point and requirement map")
    for name in CORE:
        add(ROOT / name, "project/" + name, "runtime/training dependency or frozen source identity")
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        add(path, "project/tests/" + path.name, "mechanics, checkpoint and submission-profile tests")
    for name in TOOLS:
        add(ROOT / "tools" / name, "project/tools/" + name, "final-study evaluation/review or package verification")
    for name in REPORTS:
        add(ROOT / "reports" / name, "project/reports/" + name, "final findings, demo selection, script or figure")

    final = ROOT / "artifacts/mitra_final"
    for part in ("dataset", "source_snapshot", "results"):
        for path in sorted((final / part).rglob("*")):
            if not path.is_file() or path.suffix not in (".json", ".py"):
                continue
            if "process" in path.name or path.name == "completion.json":
                continue
            add(path, "project/" + path.relative_to(ROOT).as_posix(), "one final corpus, frozen producer or raw final-study evidence")

    # Each reported final run keeps its fixed-final checkpoint, optimizer state,
    # training history and metadata. Duplicated minimum-loss weights are omitted.
    runs = [(name, 42) for name in ARCHITECTURES]
    runs += [(name, seed) for name in ("transformer_2", "embedding_only", "single_attention", "multi_attention") for seed in (7, 123)]
    for architecture, seed in runs:
        directory = final / f"checkpoints/{architecture}_seed{seed}"
        for name in ("latest.pt", "history.json", "manifest.json"):
            path = directory / name
            add(path, "project/" + path.relative_to(ROOT).as_posix(), "fixed-final checkpoint/history for a reported final run")
    pilot = ROOT / "artifacts/checkpoints/pilot_overfit"
    for name in ("best.pt", "manifest.json", "history.json"):
        path = pilot / name
        add(path, "project/" + path.relative_to(ROOT).as_posix(), "separate small character teaching pilot")
    for name in ("Tiny_Mitra_Final_v7.pptx", "Mitra_Final_Presenter_Guide.md"):
        add(REPO / "presentations/mitra/output" / name, "presentation/" + name, "current final presentation; one copy")

    output.mkdir(parents=True)
    entries = {}
    packaged_sources = {source.resolve(): name for name, (source, _) in selected.items()}
    for target, (source, reason) in sorted(selected.items()):
        destination = output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if target == "presentation/Mitra_Final_Presenter_Guide.md":
            # The working guide has clickable machine-local references. Make
            # included sources relative; keep optional historical sources in GitHub.
            def portable_link(match):
                reference = Path(match.group(1)).resolve()
                if reference in packaged_sources:
                    relative = os.path.relpath(packaged_sources[reference], Path(target).parent).replace("\\", "/")
                    return "](" + quote(relative, safe="/") + ")"
                if reference.is_relative_to(REPO):
                    relative = reference.relative_to(REPO).as_posix()
                    if reference.is_file():
                        return "](https://github.com/Nagaraj17/ai-learning-lab/blob/master/" + quote(relative, safe="/") + ")"
                    return "](https://github.com/Nagaraj17/ai-learning-lab/tree/master/" + quote(reference.parent.relative_to(REPO).as_posix(), safe="/") + ")"
                raise ValueError("Unmapped machine-local presenter link: " + str(reference))
            content = destination.read_text(encoding="utf-8")
            content = re.sub(r"\]\(<(C:/[^>]+)>\)", portable_link, content)
            destination.write_text(content, encoding="utf-8")
            reason += "; machine-local links rewritten for the portable copy"
        entries[target] = {"sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "bytes": destination.stat().st_size, "reason": reason}
    protocol = json.loads((final / "results/protocol.json").read_text(encoding="utf-8"))
    manifest = {
        "format": "tiny-mitra-clean-submission-v1", "source_hash": FROZEN_SOURCE,
        "dataset_hash": protocol["dataset_hash"], "primary_seed": 42,
        "primary_selection": "latest.pt at update 4000", "demo_model": "final_seed7/transformer_2",
        "ui_command": "python app.py --submission --port 7860", "final_runs": 16,
        "teaching_pilots": 1, "files": entries,
        "excluded": ["earlier studies and their datasets/checkpoints", "minimum-loss checkpoint duplicates", "older decks and duplicate slide copies", "execution logs/queues/process IDs", "virtual environments and caches", "private reference materials", "optional learning/Colab notebooks"],
        "limits": "Author preparation; no presenter mastery, rehearsal, latest browser visual approval or actual Colab/GPU execution is certified.",
    }
    (output / "SUBMISSION_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, "tiny-mitra-submission/" + path.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        for name, row in entries.items():
            if hashlib.sha256(bundle.read("tiny-mitra-submission/" + name)).hexdigest() != row["sha256"]:
                raise ValueError("Archive hash mismatch: " + name)
    result = {"folder": str(output), "archive": str(archive), "files": len(entries) + 1, "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "status": "packaged; independent extraction/runtime verification still required"}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    package(args.output, args.archive)
