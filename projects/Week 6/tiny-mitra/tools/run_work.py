"""Bounded local work queue. Each stage must succeed before the next begins.

Inspect artifacts/results/work_process.json and the process before starting a
second copy. The scheduled chat continuation performs the later evidence review.
"""
import subprocess
import sys
from pathlib import Path
from train import atomic_json


def main():
    Path("artifacts/results").mkdir(parents=True, exist_ok=True)
    commands = [
        ["-m", "tools.pilot"],
        ["experiments.py", "--steps", "1000", "--limit", "12"],
        ["experiments.py", "--stage", "investigate", "--limit", "8"],
        ["tools/render_results.py"],
        ["tools/prepare_notebooks.py"],
    ]
    for index, command in enumerate(commands):
        atomic_json("artifacts/results/work_queue.json", {"stage": index, "command": command, "status": "running"})
        outcome = subprocess.run([sys.executable, *command])
        if outcome.returncode:
            atomic_json("artifacts/results/work_queue.json", {"stage": index, "command": command, "status": "failed", "exit_code": outcome.returncode})
            raise SystemExit(outcome.returncode)
    atomic_json("artifacts/results/work_queue.json", {"status": "execution_finished; evidence review still required", "stages": len(commands)})


if __name__ == "__main__":
    main()
