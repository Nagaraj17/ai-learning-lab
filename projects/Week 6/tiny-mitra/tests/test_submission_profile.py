"""The compact submission must not advertise checkpoints it does not ship."""
from app import MODEL_OPTIONS, SUBMISSION_MODEL_OPTIONS, build_app


def test_submission_profile_preserves_full_ui_but_limits_submission_choices():
    assert len(MODEL_OPTIONS) == 28
    assert len(SUBMISSION_MODEL_OPTIONS) == 11
    assert "learning_pilot" in SUBMISSION_MODEL_OPTIONS
    assert not any(name.startswith("assignment_v2/") for name in SUBMISSION_MODEL_OPTIONS)
    demo = build_app(submission=True)
    components = demo.config["components"]
    picker = next(row["props"] for row in components if row["props"].get("label") == "Trained models/checkpoints to compare")
    assert [choice[1] for choice in picker["choices"]] == list(SUBMISSION_MODEL_OPTIONS)
    assert picker["value"] == ["final_seed7/transformer_2"]
    history = next(row["props"] for row in components if row["props"].get("label") == "Earlier character studies and their findings")
    assert history["visible"] is False
