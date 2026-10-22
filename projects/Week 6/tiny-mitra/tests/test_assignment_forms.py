from assignment_data import create_dataset, instructor_examples, slot_values, validate
from assignment_evaluate import counts_for_note


def test_assignment_forms_have_exact_expected_transformations():
    examples = instructor_examples()
    assert examples[0]["source"] == "PATIENT John Smith VISITED Dr Jones ON January 12"
    assert examples[0]["target"] == "PATIENT [NAME] VISITED [DOCTOR] ON [DATE]"
    assert examples[4]["target"] == "PATIENT [NAME] LIVES [ADDRESS]"
    assert examples[7]["source"] == "PATIENT Olivia Martinez MRN 991723 DIAGNOSIS NSCLC"
    assert examples[7]["target"] == "PATIENT [NAME] MRN [MRN] DIAGNOSIS NSCLC"
    for record in examples:
        result = counts_for_note(record, record["target"])
        assert result["fn"] == result["fp"] == 0
        assert result["clinical_correct"] == result["clinical_support"]


def test_wrong_doctor_date_positions_and_clinical_corruption_are_not_credited():
    record = instructor_examples()[0]
    result = counts_for_note(record, "PATIENT [NAME] VISITED [DATE] ON [DOCTOR]")
    assert result["tp"] == 1 and result["fp"] == 2 and result["fn"] == 2
    clinical = instructor_examples()[8]
    wrong = counts_for_note(clinical, "PATIENT [NAME] DIAGNOSIS ASTHMA BIOMARKER EGFR TREATMENT DrugA")
    assert wrong["clinical_correct"] == 2 and wrong["clinical_support"] == 3


def test_assignment_dataset_covers_all_source_patterns_without_breaking_holdouts():
    records = create_dataset()
    report = validate(records)
    assert report["counts"]["train"] == 700 and report["counts"]["validation"] == 100
    assert report["trained_templates"] == list(range(10))
    assert report["maximum_full_tokens"] <= 257
    assert not any("Olivia Martinez" in row["source"] for row in records if row["split"] == "train")
