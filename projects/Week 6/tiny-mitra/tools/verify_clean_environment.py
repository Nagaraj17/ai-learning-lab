"""Verify a genuinely isolated Python environment, tests and real CPU inference.

Invoke with .venv-repro/Scripts/python.exe. It does not start training or a server.
"""
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    prefix = Path(sys.prefix).resolve()
    assert prefix != Path(sys.base_prefix).resolve(), "Use an isolated virtual environment"
    configuration = (prefix / "pyvenv.cfg").read_text(encoding="utf-8").lower()
    assert "include-system-site-packages = false" in configuration
    packages = {}
    for name in ("torch", "numpy", "matplotlib", "gradio", "pytest"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(prefix), f"Inherited dependency: {name}"
        packages[name] = {"version": importlib.metadata.version(name), "path": str(path)}
    assert importlib.util.find_spec("transformers") is None, "Unexpected pretrained-model dependency"
    check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "--junitxml=artifacts/results/clean-environment-tests.xml"], check=True)
    import torch
    torch.set_num_threads(2)
    from app import build_app, compare
    # Construct the actual UI and verify its real checkpoint route. This does
    # not require the separately running server or assert unseen-note quality.
    demo = build_app()
    assert demo.config["components"]
    source = "PATIENT John Smith VISITED Dr Jones ON January 12"
    caption, outputs = compare(source, ["task_tokens/transformer_2"])
    actual = outputs["task_tokens/transformer_2"]
    assert actual["text"] == "PATIENT [NAME] VISITED [DOCTOR] ON [DATE]" and actual["protocol_valid"]
    assert "checkpoint update: 4000" in caption and "training seed: 42" in caption
    from train import atomic_json
    atomic_json("artifacts/results/clean_environment.json", {"passed": True, "python": sys.version, "environment": str(prefix), "system_site_packages": False, "packages": packages, "pip_check": check.stdout.strip(), "test_exit_code": tests.returncode, "gradio_components": len(demo.config["components"]), "actual_generation": actual, "scope": "fresh local Windows CPU installation and real checkpoint route; not an actual Colab/GPU or complete-browser verification"})
    print("Clean isolated environment: dependency check, full tests, Gradio construction and actual checkpoint inference passed")


if __name__ == "__main__":
    report = Path("artifacts/results/clean_environment.json")
    try:
        main()
    except Exception as error:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"passed": False, "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n", encoding="utf-8")
        raise
