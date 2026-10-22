"""Wait for already launched Windows workers, then verify/package local results.

The wait reads process handles, never clicks the desktop or prevents screen lock.
Do not launch a second finisher; inspect its saved process metadata first.
"""
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from train import atomic_json


def process_running(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE only
    if not handle:
        return False
    try:
        kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        return kernel.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT
    finally:
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel.CloseHandle(handle)


def run_queue():
    # The original repeat/final workers have completed and passed their audit.
    # Earlier studies are complete. Monitor only the live final workers and use
    # completion artifacts first so a recycled historical PID cannot block us.
    workers = [json.loads(Path("artifacts/mitra_final/results/process.json").read_text(encoding="utf-8-sig"))]
    workers.append(json.loads(Path("artifacts/mitra_final/results/repeats_process.json").read_text(encoding="utf-8-sig")))
    summaries = [Path("artifacts/mitra_final/results/summary.json"), Path("artifacts/mitra_final/results/repeats_summary.json")]
    while True:
        pending = []
        for worker, summary_path in zip(workers, summaries):
            rows = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
            if len(rows) == 8 and all(row["updates"] == 4000 for row in rows): continue
            if not process_running(worker["pid"]): raise RuntimeError(f"Worker exited before completion: {summary_path}")
            pending.append(worker["pid"])
        if not pending: break
        atomic_json("artifacts/results/finish_queue.json", {"status": "waiting for existing workers", "pids": pending})
        time.sleep(30)
    # Completion must be checked from artifacts as well as process termination.
    repeated = json.loads(Path("artifacts/results/repeats_summary.json").read_text())
    final = json.loads(Path("artifacts/results/final_test/summary.json").read_text())
    assignment = json.loads(Path("artifacts/assignment_v2/results/summary.json").read_text())
    task = json.loads(Path("artifacts/task_tokens/results/summary.json").read_text())
    submission = json.loads(Path("artifacts/mitra_final/results/summary.json").read_text())
    submission_repeats = json.loads(Path("artifacts/mitra_final/results/repeats_summary.json").read_text())
    if len(repeated) != 8 or len(final) != 32 or len(assignment) != 8 or len(task) != 8 or len(submission) != 8 or len(submission_repeats) != 8 or any(row["updates"] != 2000 for row in assignment + task) or any(row["updates"] != 4000 for row in submission + submission_repeats):
        raise RuntimeError("Worker exited without complete expected artifacts; inspect logs")
    commands = [["-m", "tools.audit_original_study"], ["-m", "tools.audit_assignment_study"], ["-m", "tools.audit_task_study", "--root", "artifacts/task_tokens"], ["-m", "tools.audit_task_study"], ["-m", "tools.audit_final_repeats"], ["-m", "tools.clinical_acceptance"], ["-m", "tools.clinical_acceptance", "--seed", "7"], ["-m", "tools.clinical_acceptance", "--seed", "123"], ["-m", "tools.select_demo_checkpoint"], ["-m", "tools.final_partitions"], ["-m", "tools.audit_final_partitions"], ["-m", "tools.evaluate_selected_demo"], ["-m", "tools.investigate_task"], ["-m", "tools.audit_task_investigations"], ["-m", "pytest", "-q", "--junitxml=artifacts/results/mechanics-tests.xml"], ["-m", "tools.pilot"], ["-m", "tools.verify_full_ui", "--require-assignment", "--require-task", "--require-final-partitions"], ["-m", "tools.measure_latency"], ["-m", "tools.training_exposure"], ["-m", "tools.render_final_comparison"], ["tools/render_results.py"], ["-m", "tools.build_fallback"], ["tools/prepare_notebooks.py"], ["-m", "tools.package_demo"]]
    for command in commands:
        if command == ["-m", "tools.investigate_task"] and Path("artifacts/mitra_final/results/investigations/audit.json").exists():
            continue  # Already executed; the following audit checks its identity.
        if command == ["-m", "tools.pilot"] and Path("artifacts/results/pilot.json").exists():
            continue
        atomic_json("artifacts/results/finish_queue.json", {"status": "running", "command": command})
        subprocess.run([sys.executable, *command], check=True)
    subprocess.run([sys.executable, "-m", "tools.verify_demo_bundle"], check=True)
    # The presentation source refuses incomplete/unaudited data. Rendering can
    # proceed unattended, while final visual/content review stays an open gate.
    workspace = Path(__file__).resolve().parents[4]
    node = Path("C:/Users/Nagar/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe")
    deck_source = workspace / "presentations/mitra/.build/build_final.mjs"
    atomic_json("artifacts/results/finish_queue.json", {"status": "rendering presentation from audited final evidence; review remains pending", "source": str(deck_source)})
    subprocess.run([str(node), str(deck_source)], cwd=workspace, check=True)
    quality = json.loads(Path("artifacts/mitra_final/results/audit.json").read_text())
    clinical = json.loads(Path("artifacts/mitra_final/results/clinical_acceptance.json").read_text())
    demo = json.loads(Path("reports/DEMO_SELECTION.json").read_text())
    atomic_json("artifacts/results/finish_queue.json", {"status": "local verification/package execution and presentation render complete; evidence/content/visual review still required", "primary_seed42_authored_quality_gate": quality["demo_ready"], "development_selected_demo_gate": demo["passed"], "selected_demo_model": demo["selected"]["model_id"] if demo["selected"] else None, "clinical_acceptance_passed": clinical["passed"], "presentation": str(workspace / "presentations/mitra/output/Tiny_Mitra_Final_v6.pptx"), "scope": "Engineering execution does not award learner mastery; held-out results require interpretation"})


def main():
    # A request belongs only to this worker thread and disappears when it exits.
    # It allows screen lock/display sleep while keeping authorized CPU work alive.
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.SetThreadExecutionState.argtypes = (ctypes.c_uint32,)
    kernel.SetThreadExecutionState.restype = ctypes.c_uint32
    active = bool(kernel.SetThreadExecutionState(0x80000001))  # continuous + system required
    atomic_json("artifacts/results/power_request.json", {"active": active, "scope": "temporary system-awake request for this queue; display and lock remain available", "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        run_queue()
    finally:
        kernel.SetThreadExecutionState(0x80000000)  # release only this thread's request
        atomic_json("artifacts/results/power_request.json", {"active": False, "ended_utc": datetime.now(timezone.utc).isoformat(), "scope": "queue thread released its temporary request"})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        atomic_json("artifacts/results/finish_queue.json", {"status": "failed; inspect logs and preserve existing artifacts", "error": f"{type(error).__name__}: {error}"})
        raise
