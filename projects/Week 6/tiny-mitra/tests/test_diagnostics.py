"""Category details must sum to the existing global replacement counts."""
from data import render
from diagnostics import category_precision
from evaluate import note_counts


def test_category_precision_includes_wrong_types_and_unassociated_placeholders():
    record = render([("PATIENT", "Jo"), ("MRN", "12345"), ("DIAGNOSIS", "NSCLC")])
    record.update(id="case", no_identifiers=False)
    prediction = "[NAME] PATIENT [NAME] MRN [NAME] DIAGNOSIS NSCLC"
    global_counts = note_counts(record, prediction)
    categories = category_precision([record], [{"id": "case", "generated": prediction}])
    assert categories["PATIENT"]["tp"] == 1
    assert categories["MRN"]["fp"] == 1 and categories["MRN"]["fn"] == 1
    assert categories["UNASSOCIATED"]["fp"] == 1
    assert sum(row["tp"] for row in categories.values()) == global_counts["tp"]
    assert sum(row["fp"] for row in categories.values()) == global_counts["fp"]
    assert sum(row["fn"] for row in categories.values()) == global_counts["fn"]
