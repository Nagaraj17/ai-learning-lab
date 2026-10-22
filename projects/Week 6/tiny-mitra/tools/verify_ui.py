"""Exercise the running Gradio API; a browser check is recorded separately."""
import json
from pathlib import Path
from gradio_client import Client


def main():
    client = Client("http://127.0.0.1:7860", verbose=False)
    source = "PATIENT John Lee DIAGNOSIS NSCLC TREATMENT DrugA"
    markdown, traces = client.predict(source, ["embedding_only"], "greedy", 0.7, api_name="/compare")
    assert "embedding_only" in traces and traces["embedding_only"]["tokens"]
    assert "checkpoint update" in markdown
    evidence = client.predict(api_name="/evidence")
    assert len(evidence) == 3
    missing_error = False
    try:
        client.predict("embedding_only", source, "", 0, 0, api_name="/inspect")
    except Exception as error:
        missing_error = "no attention" in str(error).lower()
    assert missing_error
    result = {"passed": True, "scope": "API: real embedding checkpoint generation, token trace, saved evidence response, expected no-attention error; full-model/head UI verification still pending", "generated": traces["embedding_only"]["text"], "termination": traces["embedding_only"]["termination"], "model_count_in_evidence": len(evidence[0].get("data", []))}
    Path("artifacts/results/ui_api_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
