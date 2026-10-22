"""Exercise incremental Gradio events and saved test metrics on localhost."""
import json
from pathlib import Path
from gradio_client import Client
from generate import generate
from task_runtime import load_task_checkpoint
from train import atomic_json


def main():
    client = Client("http://127.0.0.1:7860", verbose=False, httpx_kwargs={"timeout": 60.0, "trust_env": False})
    source = "PATIENT John Smith VISITED Dr Jones ON January 12"
    choice = "final_seed7/transformer_2"
    job = client.submit(source, choice, "greedy", 0.7, api_name="/live_tokens")
    frames = list(job)
    final = job.result()
    model, tokenizer, _ = load_task_checkpoint("artifacts/mitra_final/checkpoints/transformer_2_seed7/latest.pt")
    expected = generate(model, tokenizer, source)
    assert final[1] == expected
    assert len(frames) > 1 and all(len(frame[1]["tokens"]) <= len(expected["tokens"]) for frame in frames)
    assert all(expected["tokens"][:len(frame[1]["tokens"])] == frame[1]["tokens"] for frame in frames)
    assert "training seed 7" in final[0]
    table, status = client.predict(api_name="/task_token_metrics")
    assert len(table["data"]) == 32 and "correct output prefix" in status
    findings = client.predict(api_name="/final_findings")
    assert all(f"H{i}:" in findings for i in range(1, 10))
    atomic_json("artifacts/results/live_ui_api_check.json", {"passed": True, "model_id": choice, "source": source, "streamed_frames": len(frames), "frame_lengths": [len(frame[1]["tokens"]) for frame in frames], "final": expected, "token_metric_rows": len(table["data"]), "reviewed_hypotheses": 9, "scope": "actual incremental Gradio API events match frozen generate.py probabilities/text/stopping; final metrics and reviewed findings loaded; browser inspection separate"})
    print(f"Live API emitted{len(frames)} matching frames;32 token-metric rows and9 reviewed findings")


if __name__ == "__main__":
    main()
