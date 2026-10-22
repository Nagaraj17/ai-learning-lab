"""Check local PyTorch primitives; this is not Tiny Mitra training.

Run with a candidate Python interpreter and --output <report.json>.
No packages are installed and no GPU or full project environment is assumed.
"""

import argparse
import importlib.util
import json
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU primitives and parameter serialization only; no Mitra model",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "checks": {},
    }
    try:
        import torch

        torch.set_num_threads(2)
        report.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available())
        report["optional_packages_present"] = {
            name: importlib.util.find_spec(name) is not None
            for name in ("numpy", "gradio", "pytest", "matplotlib")
        }
        # Two numbers form a row vector. Matmul produces 1*2 + 2*1 = 4.
        x = torch.tensor([[1.0, 2.0]])
        weight = torch.nn.Parameter(torch.tensor([[2.0], [1.0]]))
        optimizer = torch.optim.SGD([weight], lr=0.01)
        prediction = x @ weight
        loss = prediction.square().mean()
        optimizer.zero_grad()
        loss.backward()
        # d(prediction^2)/d(weight) = 2*4*[1, 2] = [8, 16].
        report["checks"]["forward_and_gradient"] = bool(
            torch.equal(prediction.detach(), torch.tensor([[4.0]]))
            and torch.equal(weight.grad, torch.tensor([[8.0], [16.0]]))
        )
        optimizer.step()
        report["checks"]["optimizer_reduces_toy_loss"] = bool((x @ weight).square().mean() < loss)
        # Save a state dictionary, then load on CPU into a fresh primitive.
        layer = torch.nn.Linear(2, 1)
        layer.eval()
        expected = layer(x).detach()
        with tempfile.TemporaryDirectory(prefix="mitra-env-") as temporary:
            checkpoint = Path(temporary) / "primitive.pt"
            torch.save(layer.state_dict(), checkpoint)
            restored = torch.nn.Linear(2, 1)
            restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            restored.eval()
            report["checks"]["cpu_state_roundtrip"] = bool(torch.equal(expected, restored(x).detach()))
        report["passed"] = all(report["checks"].values())
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
