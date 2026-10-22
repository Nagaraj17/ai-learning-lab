"""Verify the extracted submission; no training and no browser interaction.

Run this from the extracted project. Its temporary local API server is closed
after verification. Visual inspection and presenter rehearsal remain separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
from urllib.parse import unquote

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]


def verify(report_path):
    package = ROOT.parent
    manifest = json.loads((package / "SUBMISSION_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "tiny-mitra-clean-submission-v1"
    for name, row in manifest["files"].items():
        path = (package / name).resolve()
        assert path.is_relative_to(package.resolve()), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], name
        assert not re.search(r"(^|/)(?:\.venv[^/]*|__pycache__|private|assignment_v2|task_tokens_broad|task_tokens_pilot)(/|$)|\.(?:log|zip)$", name), name
    assert not (ROOT / "artifacts/assignment_v2").exists()
    assert len(list((package / "presentation").glob("*.pptx"))) == 1
    checked_links = 0
    for document in package.rglob("*.md"):
        content = document.read_text(encoding="utf-8")
        for href in re.findall(r"\]\(([^)]+)\)", content):
            href = href.strip("<>")
            assert not re.match(r"[A-Za-z]:[/\\]", href), (document, href)
            if href.startswith(("http://", "https://", "#")):
                continue
            target = (document.parent / unquote(href.split("#", 1)[0])).resolve()
            assert target.exists(), (document, href)
            checked_links += 1

    from app import SUBMISSION_MODEL_OPTIONS, build_app
    from config import ARCHITECTURES
    from data import digest
    from generate import generate
    from task_runtime import load_task_checkpoint, task_source_hash
    import app

    torch.set_num_threads(2)
    assert task_source_hash() == manifest["source_hash"]
    final = ROOT / "artifacts/mitra_final"
    records = json.loads((final / "dataset/examples.json").read_text(encoding="utf-8"))
    assert len(records) == 1000 and digest(records) == manifest["dataset_hash"]
    primary = json.loads((final / "results/protocol.json").read_text(encoding="utf-8"))
    assert primary["source_hash"] == manifest["source_hash"]
    assert hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in primary["producer_files"])).hexdigest() == primary["data_producer_source_hash"]

    runs = [(name, 42) for name in ARCHITECTURES]
    runs += [(name, seed) for name in ("transformer_2", "embedding_only", "single_attention", "multi_attention") for seed in (7, 123)]
    for architecture, seed in runs:
        path = final / f"checkpoints/{architecture}_seed{seed}/latest.pt"
        model, tokenizer, saved = load_task_checkpoint(path)
        assert model.config.architecture == architecture
        assert saved["step"] == 4000 and saved["protocol"]["seed"] == seed
        assert saved["dataset_hash"] == manifest["dataset_hash"]
        assert saved["source_hash"] == manifest["source_hash"]
    assert len(list((ROOT / "artifacts").rglob("*.pt"))) == 17

    selected, tokenizer, metadata = app.selected_model("final_seed7/transformer_2")
    acceptance = json.loads((final / "results/transformer_2_seed7_acceptance.json").read_text(encoding="utf-8"))
    authored = []
    for row in acceptance["outputs"]:
        generated = generate(selected, tokenizer, row["source"])
        assert generated["text"] == row["generated"] == row["expected"]
        assert generated["termination"] == row["termination"]
        authored.append(row["id"])
    assert len(authored) == 11

    # These are API checks, not a claim about the browser's visual appearance.
    demo = build_app(submission=True)
    pickers = [row["props"] for row in demo.config["components"] if row["props"].get("label") == "Trained models/checkpoints to compare"]
    assert len(pickers) == 1
    assert [row[1] for row in pickers[0]["choices"]] == list(SUBMISSION_MODEL_OPTIONS)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    try:
        demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=port, share=False, inbrowser=False, prevent_thread_lock=True, quiet=True)
        from gradio_client import Client
        client = Client(f"http://127.0.0.1:{port}", verbose=False, httpx_kwargs={"timeout": 60.0, "trust_env": False})
        source = "PATIENT John Smith VISITED Dr Jones ON January 12"
        _, outputs = client.predict(source, list(SUBMISSION_MODEL_OPTIONS), "greedy", 0.7, api_name="/compare")
        assert set(outputs) == set(SUBMISSION_MODEL_OPTIONS)
        _, pilot = client.predict("PATIENT Jo TREATMENT DrugA", ["learning_pilot"], "greedy", 0.7, api_name="/compare")
        assert pilot["learning_pilot"]["text"] == "PATIENT [NAME] TREATMENT DrugA"
        _, pilot_b = client.predict("PATIENT Jo TREATMENT DrugB", ["learning_pilot"], "greedy", 0.7, api_name="/compare")
        assert pilot_b["learning_pilot"]["text"] == "PATIENT [NAME] TREATMENT DrugB"
        _, phone = client.predict("PATIENT Alice White PHONE 5551234567", ["task_tokens/transformer_2", "final_seed7/transformer_2"], "greedy", 0.7, api_name="/compare")
        assert phone["final_seed7/transformer_2"]["text"] == "PATIENT [NAME] PHONE [PHONE]"
        assert phone["task_tokens/transformer_2"]["text"] != phone["final_seed7/transformer_2"]["text"]
        probe = "PATIENT John Smith RECEIVED DrugA"
        _, attention = client.predict("task_tokens/transformer_2", probe, "PATIENT [NAME] RECEIVED ", 0, 0, api_name="/inspect")
        assert abs(sum(attention["attention"]) - 1) < 1e-5
        _, intervention = client.predict("task_tokens/transformer_2", probe, 0, 0, api_name="/intervene")
        _, restored = client.predict(probe, ["task_tokens/transformer_2"], "greedy", 0.7, api_name="/compare")
        assert restored["task_tokens/transformer_2"]["text"] == intervention["baseline"]["text"]
        for metric in ("recall", "conditional_loss", "clinical_preservation"):
            _, heads = client.predict(metric, "task_tokens", api_name="/head_evidence")
            assert len(heads["effects"]) == 8
        table, _, _ = client.predict(api_name="/task_evidence")
        assert len(table["data"]) == 8
        repeats, _ = client.predict(api_name="/task_repeat_evidence")
        assert len(repeats["data"]) == 8
        partitions, _ = client.predict(api_name="/task_final_partitions")
        assert len(partitions["data"]) == 32
        demo_partitions, _ = client.predict(api_name="/demo_final_partitions")
        assert len(demo_partitions["data"]) == 4
        tokens, _ = client.predict(api_name="/task_token_metrics")
        assert len(tokens["data"]) == 32
        context, context_raw, _ = client.predict(api_name="/context_evidence")
        assert len(context["data"]) == 4
        for row in context_raw["models"]:
            checkpoint = final / f"checkpoints/{row['architecture']}_seed42/latest.pt"
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row["checkpoint_sha256"]
        findings = client.predict(api_name="/final_findings")
        assert all(f"H{i}:" in findings for i in range(1, 10))
        authored_text = client.predict("transformer_2", api_name="/task_acceptance")
        assert "instructor_10" in authored_text
        job = client.submit(source, "final_seed7/transformer_2", "greedy", 0.7, api_name="/live_tokens")
        frames = list(job)
        actual = job.result()[1]
        expected = generate(selected, tokenizer, source)
        assert actual == expected and len(frames) > 1
        assert all(expected["tokens"][:len(frame[1]["tokens"])] == frame[1]["tokens"] for frame in frames)
    finally:
        demo.close()
        plt.close("all")

    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    assert tests.returncode == 0, tests.stdout + tests.stderr
    result = {
        "passed": True, "scope": "fresh extracted submission; hashes, 16 final checkpoint identities, 11 authored generations, all 11 exposed model routes, pilot pair, PHONE failure/success, attention, head restoration, saved tables, incremental API generation and packaged tests",
        "files_verified": len(manifest["files"]), "final_checkpoints": len(runs),
        "local_document_links": checked_links,
        "teaching_pilots": 1, "ui_choices": len(SUBMISSION_MODEL_OPTIONS),
        "authored_correct": len(authored), "streamed_frames": len(frames),
        "primary_partition_rows": 32, "demo_partition_rows": 4,
        "token_metric_rows": 32, "source_hash": task_source_hash(),
        "dataset_hash": manifest["dataset_hash"], "tests": tests.stdout,
        "python": sys.version, "torch": str(torch.__version__), "device": "cpu",
        "limitations": "Author engineering verification only. Browser appearance, actual cloud runtime, presenter teach-back and timed rehearsal are not certified.",
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "tests"}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/results/submission_runtime_verification.json")
    args = parser.parse_args()
    verify(args.report)
