"""Create a self-contained local demo ZIP with real weights and a file manifest."""
import hashlib
import json
from pathlib import Path
import zipfile
from config import ARCHITECTURES

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = list(ROOT.glob("*.py")) + list(ROOT.glob("*.ipynb"))
    files += [ROOT / "README.md", ROOT / "START_HERE.md", ROOT / "requirements.txt", ROOT / "requirements-cpu-tested.txt"]
    files += list((ROOT / "tests").glob("*.py")) + list((ROOT / "tools").glob("*.py"))
    files += list((ROOT / "tools").glob("*.ps1"))
    files += [path for path in (ROOT / "reports").rglob("*") if path.is_file()]
    files += [path for path in (ROOT / "artifacts/dataset").glob("*.json")]
    files += [path for path in (ROOT / "artifacts/results").rglob("*") if path.is_file() and path.suffix in (".json", ".xml", ".csv") and "process" not in path.name and path.name not in ("work_queue.json", "finish_queue.json", "power_request.json", "demo_bundle_verification.json")]
    for architecture in ARCHITECTURES:
        directory = ROOT / f"artifacts/checkpoints/{architecture}_seed42"
        for name in ("best.pt", "latest.pt", "manifest.json", "history.json"):
            path = directory / name
            if not path.exists(): raise FileNotFoundError(path)
            files.append(path)
    for architecture in ("embedding_only", "single_attention", "multi_attention", "transformer_2"):
        for seed in (7, 123):
            for name in ("best.pt", "latest.pt", "manifest.json", "history.json"):
                path = ROOT / f"artifacts/checkpoints/{architecture}_seed{seed}" / name
                if not path.exists(): raise FileNotFoundError("Original repeat series incomplete: " + str(path))
                files.append(path)
    files += [ROOT / "artifacts/checkpoints/transformer_2_seed42/latest.pt", ROOT / "artifacts/checkpoints/pilot_overfit/best.pt"]
    assignment = ROOT / "artifacts/assignment_v2"
    files += list((assignment / "dataset").glob("*.json"))
    files += [path for path in (assignment / "results").rglob("*") if path.is_file() and path.suffix in (".json", ".csv", ".xml") and "process" not in path.name]
    for architecture in ARCHITECTURES:
        for name in ("best.pt", "manifest.json", "history.json"):
            path = assignment / f"checkpoints/{architecture}_seed42" / name
            if not path.exists(): raise FileNotFoundError("Assignment-format series is incomplete: " + str(path))
            files.append(path)
    task = ROOT / "artifacts/task_tokens"
    files += list((task / "dataset").glob("*.json"))
    files += [path for path in (task / "results").rglob("*") if path.is_file() and path.suffix in (".json", ".csv", ".xml") and "process" not in path.name]
    for architecture in ARCHITECTURES:
        for name in ("latest.pt", "best.pt", "manifest.json", "history.json"):
            path = task / f"checkpoints/{architecture}_seed42" / name
            if not path.exists(): raise FileNotFoundError("Task-piece series is incomplete: " + str(path))
            files.append(path)
    final = ROOT / "artifacts/mitra_final"
    files += list((final / "dataset").glob("*.json"))
    files += [path for path in (final / "results").rglob("*") if path.is_file() and path.suffix in (".json", ".csv", ".xml") and "process" not in path.name]
    files += list((final / "source_snapshot").glob("*.py"))
    for architecture in ARCHITECTURES:
        for name in ("latest.pt", "best.pt", "manifest.json", "history.json"):
            path = final / f"checkpoints/{architecture}_seed42" / name
            if not path.exists(): raise FileNotFoundError("Final single-corpus study incomplete: " + str(path))
            files.append(path)
    repeats = json.loads((final / "results/repeats_protocol.json").read_text(encoding="utf-8"))
    for architecture in repeats["architectures"]:
        for seed in repeats["seeds"]:
            for name in ("latest.pt", "best.pt", "manifest.json", "history.json"):
                path = final / f"checkpoints/{architecture}_seed{seed}" / name
                if not path.exists(): raise FileNotFoundError("Final repeat series incomplete: " + str(path))
                files.append(path)
    # Source snapshots explain old studies without silently substituting today's
    # changed producer files. They contain project code only, never book material.
    for study in (assignment, task):
        files += list((study / "source_snapshot").glob("*.py"))
    files = sorted(set(files))
    manifest = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    output = ROOT / "artifacts/bundles/tiny-mitra-demo.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, "tiny-mitra-demo/" + path.relative_to(ROOT).as_posix())
        bundle.writestr("tiny-mitra-demo/demo_manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(output) as bundle:
        for name, fingerprint in manifest.items():
            assert hashlib.sha256(bundle.read("tiny-mitra-demo/" + name)).hexdigest() == fingerprint
    print(f"Verified local demo ZIP: {len(manifest)} files, {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
