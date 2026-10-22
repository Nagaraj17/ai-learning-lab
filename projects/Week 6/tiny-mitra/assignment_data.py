"""Assignment sentence forms, version 2; the first experiment stays untouched.

This module supplies supervised examples and evaluation annotations. The model's
generation function never calls the renderer or receives a gold answer.
"""
import itertools
import json
from pathlib import Path
import random
import re
import string
from data import CLINICAL, NAME_POOLS, REDACT, digest
from tokenizer import CharacterTokenizer

PATTERNS = (
    "PATIENT {PATIENT} VISITED {DOCTOR} ON {DATE}",
    "PATIENT {PATIENT} DOB {DOB}",
    "MRN {MRN} PATIENT {PATIENT}",
    "PATIENT {PATIENT} PHONE {PHONE}",
    "PATIENT {PATIENT} LIVES {ADDRESS}",
    "PATIENT {PATIENT} DIAGNOSIS {DIAGNOSIS}",
    "MRN {MRN} PATIENT {PATIENT} DIAGNOSIS {DIAGNOSIS}",
    "PATIENT {PATIENT} MRN {MRN} DIAGNOSIS {DIAGNOSIS}",
    "PATIENT {PATIENT} DIAGNOSIS {DIAGNOSIS} BIOMARKER {BIOMARKER} TREATMENT {TREATMENT}",
    "PATIENT {PATIENT} RECEIVED {TREATMENT}",
    "TREATMENT {TREATMENT} PATIENT {PATIENT} DIAGNOSIS {DIAGNOSIS}",
    "DIAGNOSIS {DIAGNOSIS} PATIENT {PATIENT} MRN {MRN} TREATMENT {TREATMENT}",
    "PATIENT {PATIENT} ON {DATE} VISITED {DOCTOR} LIVES {ADDRESS}",
)
POOLS = {"known": NAME_POOLS["known"] + ["John Smith", "Mary Brown", "Robert Green", "Alice White", "Kevin Black", "Susan Gray"], "validation": NAME_POOLS["validation"], "test": NAME_POOLS["test"] + ["Olivia Martinez"]}
MONTHS = ("January", "March", "April", "June", "August", "October", "December")


def render_pattern(pattern, values):
    """Track source/target offsets while filling a literal-and-field pattern."""
    parts = list(string.Formatter().parse(pattern))
    source, target, identifiers, preserved, target_fields, slots = "", "", [], [], [], []
    for index, (literal, field, _, _) in enumerate(parts):
        source += literal
        target += literal
        if field is None: continue
        value = values[field]
        start, target_start = len(source), len(target)
        source += value
        target += REDACT.get(field, value)
        annotation = {"field": field, "value": value, "start": start, "end": len(source)}
        if field in REDACT: identifiers.append(dict(annotation, placeholder=REDACT[field]))
        else: preserved.append(annotation)
        target_fields.append({"field": field, "start": target_start, "end": len(target)})
        right = parts[index + 1][0] if index + 1 < len(parts) else ""
        slots.append({"field": field, "left_literal": literal, "right_literal": right})
    return {"source": source, "target": target, "identifiers": identifiers, "preserved": preserved, "target_fields": target_fields, "slots": slots, "pattern": pattern, "no_identifiers": not identifiers}


def slot_values(text, slot):
    """Evaluate field boundaries from known literal anchors, without generation."""
    left, right = slot["left_literal"], slot["right_literal"]
    boundary = r"(?<![A-Za-z0-9])" if left and left[0].isalnum() else ""
    end = r"(?=" + re.escape(right) + r"|$)" if right else r"$"
    return [match.group(1).strip() for match in re.finditer(boundary + re.escape(left) + r"(.*?)" + end, text)]


def create_dataset(seed=2027):
    rng, records, seen = random.Random(seed), [], set()
    specifications = [("train", 700, "known", "known")]
    specifications += [("validation", 25, names, layouts) for names in ("known", "validation") for layouts in ("known", "validation")]
    specifications += [(f"test_{names}_{layouts}", 50, names, layouts) for names in ("known", "test") for layouts in ("known", "test")]
    control_orders = list(itertools.permutations(CLINICAL))
    for split, count, names, layouts in specifications:
        made, attempts = 0, 0
        while made < count:
            attempts += 1
            if attempts > 100000: raise RuntimeError("Unique dataset generation exhausted its budget")
            serial = len(records)
            template = rng.choice(list(range(10)) if layouts == "known" else [10] if layouts == "validation" else [11, 12])
            values = {"PATIENT": rng.choice(POOLS[names]), "DOCTOR": "Dr Jones" if names == "known" else "Dr Moss" if names == "validation" else "Dr Reed", "DATE": f"{rng.choice(MONTHS)} {rng.randint(1,28)}", "DOB": f"{rng.choice(MONTHS)} {rng.randint(1,28)} {rng.randint(1950,2000)}", "MRN": str(480000 + serial), "PHONE": str(5550000000 + serial), "ADDRESS": f"{100 + serial} Main Street Dallas Texas", "DIAGNOSIS": rng.choice(("NSCLC", "ASTHMA", "DIABETES")), "BIOMARKER": rng.choice(("EGFR", "ALK", "NONE")), "TREATMENT": rng.choice(("DrugA", "DrugB", "DrugC", "Chemotherapy"))}
            control = made % 10 == 0
            if control:
                order = rng.choice(control_orders)
                pattern = " ".join(field + " {" + field + "}" for field in order)
            else:
                pattern = PATTERNS[template]
            item = render_pattern(pattern, values)
            if item["source"] in seen: continue
            seen.add(item["source"])
            item.update(id=f"assignment_{serial:04d}", split=split, template_id="control" if control else str(template), entity_pool=names, template_pool=layouts, dataset_version="assignment_v2")
            records.append(item)
            made += 1
    return records


def instructor_examples():
    """Authored acceptance prompts, not samples fetched by inference."""
    defaults = {"PATIENT": "John Smith", "DOCTOR": "Dr Jones", "DATE": "January 12", "DOB": "March 4 1965", "MRN": "482991", "PHONE": "5551234567", "ADDRESS": "412 Main Street Dallas Texas", "DIAGNOSIS": "NSCLC", "BIOMARKER": "EGFR", "TREATMENT": "DrugA"}
    cases = [(0, {}), (1, {"PATIENT": "Mary Brown"}), (2, {"PATIENT": "Robert Green"}), (3, {"PATIENT": "Alice White"}), (4, {"PATIENT": "Kevin Black"}), (5, {"PATIENT": "Susan Gray"}), (6, {"PATIENT": "Mary Brown"}), (7, {"PATIENT": "Olivia Martinez", "MRN": "991723"}), (8, {}), (9, {}), (9, {"TREATMENT": "DrugB"})]
    rows = []
    for index, (template, replacements) in enumerate(cases):
        item = render_pattern(PATTERNS[template], dict(defaults, **replacements))
        item.update(id=f"instructor_{index}", split="authored_acceptance", template_id=str(template), dataset_version="assignment_v2")
        rows.append(item)
    return rows


def validate(records, context=256):
    tokenizer, counts, maximum, categories = CharacterTokenizer(), {}, 0, {}
    if len(records) != 1000 or len({row["source"] for row in records}) != 1000: raise ValueError("Expected 1,000 unique sources")
    for left in POOLS:
        for right in POOLS:
            if left != right and set(POOLS[left]) & set(POOLS[right]): raise ValueError("Entity pools overlap")
    numbers = {"MRN": set(), "PHONE": set()}
    for row in records:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
        length = len(tokenizer.encode(tokenizer.prompt(row["source"]) + row["target"] + "<EOS>"))
        maximum = max(maximum, length)
        if length - 1 > context: raise ValueError("Context overflow")
        for item in row["identifiers"] + row["preserved"]:
            if row["source"][item["start"]:item["end"]] != item["value"]: raise ValueError("Invalid span")
            categories[item["field"]] = categories.get(item["field"], 0) + 1
            slot = next(slot for slot in row["slots"] if slot["field"] == item["field"])
            if slot_values(row["target"], slot) != [REDACT.get(item["field"], item["value"])]: raise ValueError("Target/anchor policy mismatch")
            if item["field"] in numbers:
                if item["value"] in numbers[item["field"]]: raise ValueError("Identifier repeated")
                numbers[item["field"]].add(item["value"])
    trained = {int(row["template_id"]) for row in records if row["split"] == "train" and not row["no_identifiers"]}
    if trained != set(range(10)): raise ValueError("Instructor or clinical grammar missing from training")
    return {"version": "assignment_v2", "dataset_hash": digest(records), "counts": counts, "category_counts": categories, "maximum_full_tokens": maximum, "context": context, "trained_templates": sorted(trained), "policy": "replace identity spans including physician titles; retain surrounding wording and clinical field/value associations", "novelty_limitations": "control layouts shared; date values may recur; name/doctor pools separate; MRN/phone/address values unique where present"}


def save_dataset():
    root = Path("artifacts/assignment_v2/dataset")
    root.mkdir(parents=True, exist_ok=True)
    records = create_dataset()
    manifest = validate(records)
    for name, value in (("examples.json", records), ("manifest.json", manifest), ("instructor_examples.json", instructor_examples())):
        (root / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return records, manifest


if __name__ == "__main__":
    _, manifest = save_dataset()
    print(json.dumps(manifest, indent=2))
