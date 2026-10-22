"""Extract the deliverable into a fresh folder and load every real UI checkpoint."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
from train import atomic_json


def main():
    root = Path(__file__).resolve().parents[1]
    bundle_path = root / "artifacts/bundles/tiny-mitra-demo.zip"
    python = root / ".venv-repro/Scripts/python.exe"
    if not python.exists(): raise ValueError("Create and verify the isolated environment first")
    archive_base = (root / "artifacts/archive").resolve()
    archive_base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bundle-check-", dir=archive_base) as temporary:
        directory = Path(temporary).resolve()
        # Verify the exact cleanup/extraction target is a fresh descendant of
        # this project's archive folder before any recursive work occurs.
        assert directory.is_relative_to(archive_base) and directory != archive_base
        with zipfile.ZipFile(bundle_path) as bundle:
            for member in bundle.namelist():
                assert (directory / member).resolve().is_relative_to(directory), "Unsafe archive path"
            manifest = json.loads(bundle.read("tiny-mitra-demo/demo_manifest.json"))
            for name, expected in manifest.items():
                assert hashlib.sha256(bundle.read("tiny-mitra-demo/" + name)).hexdigest() == expected, name
            bundle.extractall(directory)
        extracted = directory / "tiny-mitra-demo"
        script = """
import json
from pathlib import Path
import torch
torch.set_num_threads(2)
from app import MODEL_OPTIONS, selected_model, build_app
from generate import generate
outputs=[]
source='PATIENT John Smith VISITED Dr Jones ON January 12'
for choice in MODEL_OPTIONS:
    model,tokenizer,checkpoint=selected_model(choice)
    result=generate(model,tokenizer,source)
    outputs.append({'model_id':choice,'update':checkpoint['step'],'seed':checkpoint['protocol']['seed'],'generated':result['text'],'termination':result['termination']})
assert len(outputs)==28
assert next(row for row in outputs if row['model_id']=='final_seed7/transformer_2')['generated']=='PATIENT [NAME] VISITED [DOCTOR] ON [DATE]'
assert build_app().config['components']
Path('bundle_smoke.json').write_text(json.dumps({'passed':True,'checkpoint_routes':outputs}),encoding='utf-8')
print('Fresh extraction:28checkpoint routes and Gradio construction passed')
"""
        completed = subprocess.run([str(python), "-c", script], cwd=extracted, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        smoke = json.loads((extracted / "bundle_smoke.json").read_text(encoding="utf-8"))
    atomic_json(root / "artifacts/results/demo_bundle_verification.json", {"passed": True, "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(), "manifest_files": len(manifest), "isolated_interpreter": str(python), "fresh_extraction_smoke": smoke, "scope": "local fresh extraction/hash verification/all28 matching checkpoint routes/UI construction; live browser checks and learner rehearsal are separate"})
    print(completed.stdout.strip())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        atomic_json("artifacts/results/demo_bundle_verification.json", {"passed": False, "error": f"{type(error).__name__}: {error}"})
        raise
