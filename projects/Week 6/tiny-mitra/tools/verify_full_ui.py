"""Run after all eight baseline checkpoints exist and the current app is started."""
import json
import argparse
from pathlib import Path
from gradio_client import Client
from config import ARCHITECTURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-assignment", action="store_true", help="Require all eight corrected-grammar runs before packaging")
    parser.add_argument("--require-task", action="store_true", help="Require all eight fixed-task-piece runs")
    parser.add_argument("--task-root", default="artifacts/mitra_final")
    parser.add_argument("--require-final-partitions", action="store_true")
    args = parser.parse_args()
    client = Client("http://127.0.0.1:7860", verbose=False, httpx_kwargs={"timeout": 60.0, "trust_env": False})
    source = "PATIENT John Lee DIAGNOSIS NSCLC BIOMARKER EGFR TREATMENT DrugA"
    architectures = list(ARCHITECTURES)
    markdown, outputs = client.predict(source, architectures, "greedy", 0.7, api_name="/compare")
    assert set(outputs) == set(architectures)
    assert all(outputs[name]["tokens"] for name in architectures)
    prefix = "PATIENT [NAME] DIAGNOSIS NSCLC BIOMARKER EGFR TREATMENT Drug"
    figure, attention = client.predict("transformer_2", source, prefix, 0, 0, api_name="/inspect")
    assert abs(sum(attention["attention"]) - 1) < 1e-5
    assert attention["query_index"] == len(attention["input_tokens"]) - 1
    ablation_text, ablation = client.predict("transformer_2", source, 0, 0, api_name="/intervene")
    assert "Baseline" in ablation_text and "Ablated" in ablation_text
    _, after = client.predict(source, ["transformer_2"], "greedy", 0.7, api_name="/compare")
    assert after["transformer_2"]["text"] == ablation["baseline"]["text"]
    table, curves, heads = client.predict(api_name="/evidence")
    assert len(table["data"]) >= 8
    for letter in "AB":
        _, pilot = client.predict(f"PATIENT Jo TREATMENT Drug{letter}", ["learning_pilot"], "greedy", 0.7, api_name="/compare")
        assert pilot["learning_pilot"]["text"] == f"PATIENT [NAME] TREATMENT Drug{letter}"
    for metric in ("recall", "conditional_loss", "clinical_preservation"):
        _, saved = client.predict(metric, "original_characters", api_name="/head_evidence")
        assert len(saved["effects"]) == 8
    findings = client.predict(api_name="/findings")
    assert "H7" in findings and "limitations" in findings.lower()
    partitions, partition_status = client.predict(api_name="/final_partitions")
    assert "32" in partition_status
    selection = client.predict(api_name="/checkpoint_selection")
    assert "22/22" in selection
    assignment_table, assignment_curves, assignment_status = client.predict(api_name="/assignment_evidence")
    acceptance = client.predict("transformer_2", api_name="/assignment_acceptance")
    if args.require_assignment:
        assert len(assignment_table["data"]) == 8
        assert "instructor_10" in acceptance and "Generated:" in acceptance
    assignment_source = "PATIENT John Smith VISITED Dr Jones ON January 12"
    assignment_choices = ["assignment_v2/" + name for name in (ARCHITECTURES if args.require_assignment else ("transformer_2",))]
    _, actual_assignment = client.predict(assignment_source, assignment_choices, "greedy", 0.7, api_name="/compare")
    assert set(actual_assignment) == set(assignment_choices)
    # Verify routing to the right saved weights. Correctness of its text is a
    # measured quality outcome, not something this engineering test assumes.
    from generate import generate
    from train import load_checkpoint
    trained, tokenizer, saved = load_checkpoint("artifacts/assignment_v2/checkpoints/transformer_2_seed42/best.pt")
    assert actual_assignment["assignment_v2/transformer_2"]["text"] == generate(trained, tokenizer, assignment_source)["text"]
    task_table, task_curves, task_status = client.predict(api_name="/task_evidence")
    task_outputs = {}
    repeat_outputs, task_partitions, repeat_table = {}, {"data": []}, {"data": []}
    if args.require_task:
        assert len(task_table["data"]) == 8
        choices = ["task_tokens/" + name for name in ARCHITECTURES]
        _, task_outputs = client.predict(assignment_source, choices, "greedy", 0.7, api_name="/compare")
        assert set(task_outputs) == set(choices)
        from task_runtime import load_task_checkpoint
        task_model, task_tokenizer, task_checkpoint = load_task_checkpoint(Path(args.task_root) / "checkpoints/transformer_2_seed42/latest.pt")
        task_protocol = json.loads((Path(args.task_root) / "results/protocol.json").read_text(encoding="utf-8"))
        assert task_checkpoint["step"] == task_protocol.get("steps", task_protocol.get("stage_updates"))
        assert task_outputs["task_tokens/transformer_2"]["text"] == generate(task_model, task_tokenizer, assignment_source)["text"]
        task_acceptance = client.predict("transformer_2", api_name="/task_acceptance")
        assert "instructor_10" in task_acceptance and "Generated:" in task_acceptance
        for metric in ("recall", "conditional_loss", "clinical_preservation"):
            _, task_heads = client.predict(metric, "task_tokens", api_name="/head_evidence")
            assert len(task_heads["effects"]) == 8
        # The task tokenizer has different piece boundaries from the character
        # study. Check a complete-piece conditional query and request restoration.
        task_source = "PATIENT John Smith RECEIVED DrugA"
        _, task_attention = client.predict("task_tokens/transformer_2", task_source, "PATIENT [NAME] RECEIVED ", 1, 3, api_name="/inspect")
        assert abs(sum(task_attention["attention"]) - 1) < 1e-5
        assert task_attention["query_index"] == len(task_attention["input_tokens"]) - 1
        _, task_ablation = client.predict("task_tokens/transformer_2", task_source, 1, 3, api_name="/intervene")
        _, task_restored = client.predict(task_source, ["task_tokens/transformer_2"], "greedy", 0.7, api_name="/compare")
        assert task_restored["task_tokens/transformer_2"]["text"] == task_ablation["baseline"]["text"] == generate(task_model, task_tokenizer, task_source)["text"]
        repeat_table, repeat_status = client.predict(api_name="/task_repeat_evidence")
        assert len(repeat_table["data"]) == 8 and "minibatch" in repeat_status
        for seed in (7, 123):
            choice = f"final_seed{seed}/transformer_2"
            caption, actual = client.predict(task_source, [choice], "greedy", 0.7, api_name="/compare")
            reference, token_pieces, metadata = load_task_checkpoint(Path(args.task_root) / f"checkpoints/transformer_2_seed{seed}/latest.pt")
            assert metadata["protocol"]["seed"] == seed and metadata["step"] == 4000
            assert actual[choice]["text"] == generate(reference, token_pieces, task_source)["text"]
            assert f"training seed: {seed}" in caption
            repeat_outputs[choice] = actual[choice]["text"]
        task_partitions, task_partition_status = client.predict(api_name="/task_final_partitions")
        context_table, context_raw, context_status = client.predict(api_name="/context_evidence")
        assert len(context_table["data"]) == len(context_raw["models"]) == 4
        assert "supplied output prefix" in context_status
        assert next(row for row in context_raw["models"] if row["architecture"] == "embedding_only")["identical_distributions"]
        import hashlib
        for row in context_raw["models"]:
            path = Path(args.task_root) / f"checkpoints/{row['architecture']}_seed42/latest.pt"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["checkpoint_sha256"]
            assert len(row["observations"]) == 2
            assert row["dataset_hash"] == task_checkpoint["dataset_hash"]
        if args.require_final_partitions:
            assert len(task_partitions["data"]) == 32 and "200 distinct" in task_partition_status
            demo_partitions, demo_status = client.predict(api_name="/demo_final_partitions")
            assert len(demo_partitions["data"]) == 4 and "fixed before" in demo_status
        final_findings = client.predict(api_name="/final_findings")
        assert "review" in final_findings.lower() or "H9" in final_findings
    result = {"passed": True, "scope": "API: all eight original models and available/required assignment-v2 generation, attention row, request-local intervention/restoration, saved/repeated rows, both controlled pilot outputs, three head metrics, reviewed findings, partition table, checkpoint comparison and separately labelled assignment evidence; browser/visual check separately", "generated": {name: value["text"] for name, value in outputs.items()}, "assignment_generated": {name: row["text"] for name, row in actual_assignment.items()}, "assignment_checkpoint_step": saved["step"], "assignment_rows": len(assignment_table["data"]), "checkpoint_rows": len(table["data"]), "attention_tokens": len(attention["attention"]), "final_partition_rows_so_far": len(partitions["data"])}
    result.update(task_model_rows=len(task_table["data"]), task_generated={name: row["text"] for name, row in task_outputs.items()}, quality_scope="Engineering path checks are separate from authored-prompt and held-out model-quality gates")
    result.update(final_repeat_rows=len(repeat_table["data"]), repeat_generated=repeat_outputs, task_final_partition_rows=len(task_partitions["data"]), task_attention_and_intervention_restoration_checked=args.require_task)
    Path("artifacts/results/full_ui_api_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    report_path = Path("artifacts/results/full_ui_api_check.json")
    report_path.write_text(json.dumps({"passed": False, "status": "verification running"}) + "\n", encoding="utf-8")
    try:
        main()
    except Exception as error:
        report_path.write_text(json.dumps({"passed": False, "status": "verification failed", "error": f"{type(error).__name__}: {error}"}, indent=2) + "\n", encoding="utf-8")
        raise
