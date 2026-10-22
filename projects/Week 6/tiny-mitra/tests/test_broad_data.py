"""Coverage checks target the two observed generalization failures."""
import string
from assignment_data import create_dataset, instructor_examples
from broad_data import NAME_POOLS, create_broad_dataset, validate_broad
from task_tokenizer import TaskTokenizer


def test_broader_values_keep_labels_splits_and_holdouts_but_add_missing_coverage():
    old, records = create_dataset(), create_broad_dataset()
    manifest = validate_broad(records)
    assert manifest["counts"]["train"] == 700
    assert [(row["id"], row["split"], row["target"]) for row in records] == [(row["id"], row["split"], row["target"]) for row in old]
    training = [row for row in records if row["split"] == "train"]
    assert not ({row["source"] for row in training} & {row["source"] for row in instructor_examples()})
    names = [item["value"] for row in training for item in row["identifiers"] if item["field"] == "PATIENT"]
    assert max(map(len, names)) >= 18 and "Olivia Martinez" not in names
    tokenizer = TaskTokenizer()
    observed_name_ids = {token for name in names for token in tokenizer.encode(name)}
    assert {tokenizer.token_to_id[letter] for letter in string.ascii_letters} <= observed_name_ids
    phones = [item["value"] for row in training for item in row["identifiers"] if item["field"] == "PHONE"]
    mrns = [item["value"] for row in training for item in row["identifiers"] if item["field"] == "MRN"]
    assert "5551234567" not in phones and "991723" not in mrns
    development_names = {item["value"] for row in instructor_examples() for item in row["identifiers"] if item["field"] == "PATIENT"}
    assert not development_names & set(NAME_POOLS["test"])
    assert all(len({value[position] for value in phones}) >= 8 for position in range(3, 10))
    assert all(len({value[position] for value in mrns}) >= 8 for position in range(6))
