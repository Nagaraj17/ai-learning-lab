"""Broaden input-value coverage without putting authored answers into inference.

The earlier generator used mostly short names and sequential phone/MRN values.
This follow-up keeps its1,000 pattern/split/clinical-target assignments, but
varies name lengths and every identifier digit. Instructor prompt sources and
their numerical values are reserved. This is development-driven data work;
the separate final partitions are not used to tune it.
"""
import json
from pathlib import Path
import random
from assignment_data import POOLS, create_dataset, instructor_examples, render_pattern, validate
from data import digest
from task_tokenizer import TaskTokenizer

NAME_POOLS = {
    "known": POOLS["known"] + ["Adrian Montgomery", "Isabella Chandra", "Nathaniel Brooks", "Amelia Srinivasan", "Alexander Hudson", "Charlotte Iyer", "Benjamin Macdonald", "Uma Li"],
    "validation": POOLS["validation"] + ["Luciana Campbell", "Matteo Suresh"],
    "test": POOLS["test"] + ["Evelyn Banerjee", "Vikram Patel", "Anastasia Mehta"],
}


def create_broad_dataset():
    rng, seen, used_mrn, used_phone, records = random.Random(2028), set(), set(), set(), []
    reserved = {row["source"] for row in instructor_examples()}
    for old in create_dataset():
        values = {item["field"]: item["value"] for item in old["identifiers"] + old["preserved"]}
        if "MRN" in values:
            while True:
                candidate = str(rng.randint(100000, 999999))
                if candidate not in used_mrn and candidate not in ("482991", "991723"): break
            used_mrn.add(candidate)
            values["MRN"] = candidate
        if "PHONE" in values:
            while True:
                candidate = "555" + f"{rng.randrange(10_000_000):07d}"
                if candidate not in used_phone and candidate != "5551234567": break
            used_phone.add(candidate)
            values["PHONE"] = candidate
        for attempt in range(10000):
            if "PATIENT" in values: values["PATIENT"] = rng.choice(NAME_POOLS[old["entity_pool"]])
            rendered = render_pattern(old["pattern"], values)
            if rendered["source"] not in seen and rendered["source"] not in reserved: break
        else: raise RuntimeError("Broad-source uniqueness budget exhausted")
        seen.add(rendered["source"])
        # Fresh spans/slots override the old ones; split/template identity stays.
        records.append(dict(old, **rendered, dataset_version="broad_coverage_v1"))
    return records


def validate_broad(records):
    manifest = validate(records)
    for record in records:
        for item in record["identifiers"]:
            if item["field"] == "PATIENT" and item["value"] not in NAME_POOLS[record["entity_pool"]]:
                raise ValueError("Actual patient name differs from the declared pool")
    for left in NAME_POOLS:
        for right in NAME_POOLS:
            if left != right and set(NAME_POOLS[left]) & set(NAME_POOLS[right]): raise ValueError("Name pools overlap")
    training = [row for row in records if row["split"] == "train"]
    assert not ({row["source"] for row in training} & {row["source"] for row in instructor_examples()})
    tokenizer = TaskTokenizer()
    lengths = [len(tokenizer.encode(tokenizer.prompt(row["source"]) + row["target"] + "<EOS>")) for row in records]
    manifest.update(version="broad_coverage_v1", dataset_hash=digest(records), maximum_character_full_tokens=manifest["maximum_full_tokens"], maximum_full_tokens=max(lengths), tokenizer_hash=tokenizer.fingerprint, vocabulary_size=len(tokenizer), change="Expanded name lengths and random numerical identifier digits; same pattern/split/clinical-target assignments", novelty_limitations="Controls share layouts; dates/doctor/address policy unchanged; authored development sources excluded from this stage, but some may occur in the earlier parent training corpus")
    return manifest


def save_broad_dataset(root="artifacts/task_tokens_broad"):
    root = Path(root)
    records = create_broad_dataset()
    manifest = validate_broad(records)
    (root / "dataset").mkdir(parents=True, exist_ok=True)
    for name, value in (("examples.json", records), ("manifest.json", manifest)):
        (root / "dataset" / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return records, manifest


if __name__ == "__main__":
    _, manifest = save_broad_dataset()
    print(json.dumps(manifest, indent=2))
