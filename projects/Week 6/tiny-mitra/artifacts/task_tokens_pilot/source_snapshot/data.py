"""Synthetic task data and shifted output-only training items.

Gold answers exist here for training/evaluation. Inference never calls the
renderer to obtain a replacement string. IDs, hashes and annotations make the
same saved examples reusable across all architectures.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path
import torch
from tokenizer import CharacterTokenizer

REDACT = {"PATIENT": "[NAME]", "DOCTOR": "[DOCTOR]", "DATE": "[DATE]", "DOB": "[DATE]", "MRN": "[MRN]", "PHONE": "[PHONE]", "ADDRESS": "[ADDRESS]"}
CLINICAL = ("DIAGNOSIS", "BIOMARKER", "TREATMENT")
FIELDS = tuple(REDACT) + CLINICAL
# Separate template IDs, not random holdouts from one common template pool.
TEMPLATES = (
    ("PATIENT", "DIAGNOSIS", "BIOMARKER", "TREATMENT"),
    ("PATIENT", "MRN", "DIAGNOSIS", "TREATMENT"),
    ("DOCTOR", "DATE", "PATIENT", "DIAGNOSIS"),
    ("PATIENT", "DOB", "PHONE", "ADDRESS", "TREATMENT"),
    ("TREATMENT", "PATIENT", "DIAGNOSIS", "MRN"),  # validation-only layout
    ("MRN", "DIAGNOSIS", "PATIENT", "TREATMENT"),  # final-only layouts
    ("ADDRESS", "PHONE", "PATIENT", "DOB", "TREATMENT"),
)
NAME_POOLS = {
    "known": [a + " " + b for a in ("John", "Maya", "Ravi", "Sara", "Noah", "Leah") for b in ("Lee", "Shah", "Cole", "Rao")],
    "validation": [a + " " + b for a in ("Iris", "Omar", "Luca", "Nina") for b in ("Moss", "Kent")],
    "test": [a + " " + b for a in ("Zoya", "Evan", "Asha", "Hugo") for b in ("Reed", "Park")],
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_fields(text):
    """Parse the declared synthetic grammar; preserve field/value associations.

    Multiple occurrences stay visible rather than being overwritten by a dict.
    This parser evaluates generated text; it is not the inference engine.
    """
    matches = list(re.finditer(r"(?<!\S)(" + "|".join(FIELDS) + r")(?!\S)", text))
    return [(match.group(1), text[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(text)].strip()) for index, match in enumerate(matches)]


def render(field_values):
    source, target, identifiers, preserved, boundaries = "", "", [], [], []
    for field, value in field_values:
        separator = " " if source else ""
        source += separator + field + " "
        start = len(source)
        source += value
        target += separator + field + " "
        target_start = len(target)
        replacement = REDACT.get(field, value)
        target += replacement
        boundaries.append({"field": field, "start": target_start, "end": len(target)})
        annotation = {"field": field, "value": value, "start": start, "end": len(source)}
        if field in REDACT:
            identifiers.append(dict(annotation, placeholder=REDACT[field]))
        else:
            preserved.append(annotation)
    return {"source": source, "target": target, "identifiers": identifiers, "preserved": preserved, "target_fields": boundaries}


def generate_dataset(seed=2026):
    rng, records = random.Random(seed), []
    specifications = [("train", 700, "known", "known")]
    specifications += [("validation", 25, names, templates) for names in ("known", "validation") for templates in ("known", "validation")]
    specifications += [(f"test_{names}_{templates}", 50, names, templates) for names in ("known", "test") for templates in ("known", "test")]
    for split, count, names, templates in specifications:
        made = 0
        while made < count:
            serial = len(records)
            template = rng.choice(list(range(4)) if templates == "known" else ([4] if templates == "validation" else [5, 6]))
            name = rng.choice(NAME_POOLS[names])
            doctor = ("Dr Alpha" if names == "known" else "Dr Beta" if names == "validation" else "Dr Gamma")
            values = {
                "PATIENT": name, "DOCTOR": doctor,
                "DATE": f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
                "DOB": f"{rng.randint(1950,2000)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
                "MRN": str(70000000 + serial), "PHONE": f"555-{serial // 10000:03d}-{serial % 10000:04d}",
                "ADDRESS": f"{100 + serial} Oak St",
                "DIAGNOSIS": rng.choice(("NSCLC", "ASTHMA", "DIABETES")),
                "BIOMARKER": rng.choice(("EGFR", "ALK", "NONE")),
                "TREATMENT": rng.choice(("DrugA", "DrugB", "DrugC")),
            }
            # Ten percent preservation controls have no identifiers to redact.
            control = made % 10 == 0
            fields = CLINICAL if control else TEMPLATES[template]
            item = render([(field, values[field]) for field in fields])
            if any(item["source"] == existing["source"] for existing in records):
                # Only 27 combinations for the short control grammar: add a
                # clinical numeric suffix so control records remain unique.
                if control:
                    item = render([(field, values[field] + (f" dose{serial}" if field == "TREATMENT" else "")) for field in fields])
                else:
                    continue
            item.update(id=f"example_{serial:04d}", split=split, template_id="control" if control else str(template), entity_pool=names, template_pool=templates, no_identifiers=control)
            records.append(item)
            made += 1
    return records


def validate_dataset(records, tokenizer, context):
    if len(records) != 1000 or len({item["source"] for item in records}) != len(records):
        raise ValueError("Expected 1,000 globally unique sources")
    counts, category_counts, lengths, unique_ids = {}, {}, {}, {"MRN": set(), "PHONE": set()}
    for item in records:
        split = item["split"]
        counts[split] = counts.get(split, 0) + 1
        full = tokenizer.prompt(item["source"]) + item["target"] + "<EOS>"
        length = len(tokenizer.encode(full))
        lengths[split] = max(lengths.get(split, 0), length)
        if length - 1 > context:
            raise ValueError(f"Context overflow for {item['id']}: {length - 1}>{context}")
        for annotation in item["identifiers"] + item["preserved"]:
            if item["source"][annotation["start"]:annotation["end"]] != annotation["value"]:
                raise ValueError("Annotation offsets are invalid")
            field = annotation["field"]
            category_counts[field] = category_counts.get(field, 0) + 1
            if field in unique_ids:
                if annotation["value"] in unique_ids[field]:
                    raise ValueError(f"Repeated {field}")
                unique_ids[field].add(annotation["value"])
        if parse_fields(item["target"]) != [(field, REDACT.get(field, value)) for field, value in parse_fields(item["source"])]:
            raise ValueError("Source/target policy mismatch")
    for first in NAME_POOLS:
        for second in NAME_POOLS:
            if first != second and set(NAME_POOLS[first]) & set(NAME_POOLS[second]):
                raise ValueError("Name pools overlap")
    return {"counts": counts, "category_counts": category_counts, "maximum_full_tokens": lengths, "dataset_hash": digest(records), "examples": len(records), "context": context, "tokenizer_hash": tokenizer.fingerprint, "policy": "replace identifier field values; preserve clinical field/value associations", "novelty": "names/doctors use separate pools; MRNs/phones globally unique; dates may recur; control layout shared; addresses unique where present"}


def training_item(record, tokenizer, context):
    prompt = tokenizer.encode(tokenizer.prompt(record["source"]))
    all_ids = prompt + tokenizer.encode(record["target"]) + [tokenizer.eos_id]
    if len(all_ids) - 1 > context:
        raise ValueError("Example exceeds context; truncation is not permitted")
    # At input position p-1 (<OUTPUT>), the target at position p is the first
    # output token. Loss targets before p are ignored; EOS is still scored.
    x, y = all_ids[:-1], all_ids[1:]
    y = [token if index >= len(prompt) - 1 else -100 for index, token in enumerate(y)]
    return x, y


def batch_items(records, tokenizer, context, device="cpu"):
    items = [training_item(record, tokenizer, context) for record in records]
    width = max(len(x) for x, _ in items)
    x = torch.full((len(items), width), tokenizer.pad_id, dtype=torch.long)
    y = torch.full_like(x, -100)
    valid = torch.zeros_like(x, dtype=torch.bool)
    for index, (inputs, targets) in enumerate(items):
        x[index, :len(inputs)] = torch.tensor(inputs)
        y[index, :len(targets)] = torch.tensor(targets)
        valid[index, :len(inputs)] = True
    return x.to(device), y.to(device), valid.to(device)


def load_dataset(path="artifacts/dataset/examples.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/dataset")
    parser.add_argument("--context", type=int, default=512)
    args = parser.parse_args()
    records, tokenizer = generate_dataset(), CharacterTokenizer()
    manifest = validate_dataset(records, tokenizer, args.context)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "examples.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
