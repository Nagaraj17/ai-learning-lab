# Topic 5 — Evaluation, Generalization, Failure Analysis, and Safety

## Learning goal

You should be able to design an evaluation that separates copying, redaction, clinical preservation, memorization, and generalization—and explain why no single accuracy number proves safety.

## 1. Define success before measuring it

Input:

```text
MRN 456789 PATIENT Olivia Martinez DIAGNOSIS ASTHMA BIOMARKER EGFR
```

Expected:

```text
MRN [MRN] PATIENT [NAME] DIAGNOSIS ASTHMA BIOMARKER EGFR
```

A good output must:

1. remove `456789`;
2. remove `Olivia Martinez`;
3. use the right placeholders;
4. retain `ASTHMA` and `EGFR`;
5. retain useful structure;
6. avoid inventing content;
7. stop correctly.

These are separate behaviours and need separate measurements.

## 2. Why token accuracy is insufficient

Suppose an output contains 20 tokens and 19 are correct. Token accuracy is 95%. If the one wrong token is the patient’s name, the output is still a privacy failure.

Class imbalance makes this worse: safe clinical and structural tokens often outnumber PHI tokens. A model that mostly copies can appear accurate.

Use token accuracy for training progress—not as the safety conclusion.

## 3. Exact match

```text
exact_match = 1 if every expected token equals every generated token
```

### Advantage

Strict, simple, and exposes any difference.

### Limitation

It treats harmless formatting differences and dangerous leakage as equally wrong. It also gives no partial credit or explanation.

Exact match should accompany targeted metrics.

## 4. Placeholder recall

Our transparent metric counts expected placeholder occurrences that appear in generated output:

```text
placeholder recall = correctly matched expected placeholders / expected placeholders
```

Expected:

```text
PATIENT [NAME] PHONE [PHONE]
```

Generated:

```text
PATIENT [NAME] PHONE 9876543210
```

One of two expected placeholders is present, so recall is 0.5.

### Limitation

Placeholder presence does not prove that the raw identifier disappeared. An output could contain both `[NAME]` and `Olivia`. We therefore also need direct leakage scans.

## 5. Placeholder precision

```text
placeholder precision = correctly expected placeholders / all generated placeholders
```

Expected:

```text
DIAGNOSIS ASTHMA BIOMARKER EGFR
```

Generated:

```text
DIAGNOSIS [NAME] BIOMARKER [NAME]
```

The model generated placeholders where none were expected. Precision is poor and clinical information was destroyed. This is **over-redaction**.

## 6. Clinical preservation

Our baseline checks whether expected clinical marker tokens such as diagnoses, biomarkers, drugs, and treatments remain present:

```text
preservation = preserved expected clinical markers / expected clinical markers
```

Expected markers: `{NSCLC, EGFR, DrugA}`  
Generated markers: `{NSCLC, DrugA}`  
Preservation: `2/3`.

### Limitation

A set-based check ignores duplicates, order, negation, and relationships. Retaining the word `NSCLC` does not prove the generated sentence retained the same meaning. It is acceptable for the synthetic baseline, not a clinical semantic guarantee.

## 7. Direct identifier leakage

For each synthetic example we know the source identifiers. Search generated output for:

- full original values;
- individual name parts;
- MRN and phone number;
- date components;
- address components;
- normalized variants where punctuation/spaces differ.

Useful metrics:

```text
leakage rate = outputs containing any source identifier / total outputs
entity leakage rate = leaked entities / total source entities
```

In de-identification, one leak is more important than many correct non-PHI tokens.

## 8. Under-redaction, over-redaction, hallucination

| Failure | Meaning | Example |
|---|---|---|
| Under-redaction | Identifier retained | `PATIENT Olivia Martinez` |
| Wrong placeholder | Identifier removed incorrectly | phone becomes `[DATE]` |
| Over-redaction | Safe content removed | `ASTHMA` becomes `[NAME]` |
| Clinical alteration | Medical fact changed | `DrugA` becomes `DrugB` |
| Hallucination | New unsupported content | New diagnosis appears |
| Structural failure | Output malformed/repeated | Endless `PATIENT PATIENT` |
| Stop failure | Missing or early `<EOS>` | Truncated note |

Every evaluation report should include concrete examples from each observed category.

## 9. Memorization versus generalization

### Memorization test

Evaluate a training example. Success may mean the model remembers it.

### Unseen entity test

Use a familiar template but names and identifiers absent from training.

### Unseen template test

Use familiar entity types in a sentence structure absent from training.

### Combined challenge

Use both unseen values and unseen structures.

Report these separately:

| Split | Entity values | Template | What it tests |
|---|---|---|---|
| Train | Seen | Seen | Fit/memorization |
| Validation | Usually seen pools | Seen | Tuning within distribution |
| Test-value | Unseen | Seen | Contextual entity generalization |
| Test-template | Seen/unseen | Unseen | Structural generalization |
| Stress | Unseen | Unseen/noisy | Robustness boundary |

## 10. Data leakage checks

Before training:

1. verify test-only names/doctors do not appear in train;
2. fit tokenizer only on training sequences;
3. ensure repeated generated MRNs do not cross splits;
4. ensure templates intended as unseen are absent from training;
5. record random seeds and split logic;
6. inspect duplicates after normalization.

If train/test separation is wrong, impressive metrics can be meaningless.

## 11. Model comparison must be fair

Compare Models A, B, and C using:

- identical data splits;
- identical tokenizer;
- identical training budget or clearly reported differences;
- identical evaluation prompts;
- identical decoding policy;
- parameter count;
- multiple random seeds when feasible.

Model A cannot use earlier tokens contextually. Model B has one causal attention head. Model C stacks full multi-head decoder blocks with FFNs and LayerNorm. Better performance should be attributed carefully: Model C may have both better architecture and more parameters.

## 12. Temperature evaluation

Use low temperature for the primary deterministic result, then stress-test sampling:

| Temperature | Runs | Purpose |
|---:|---:|---|
| 0.1 | 1 | Stable baseline |
| 0.7 | 5–10 | Moderate variability |
| 1.2 | 5–10 | Instability stress test |

Track:

- identifier leakage;
- clinical alteration;
- exact match;
- failure to stop;
- variability across repeated runs.

## 13. Attention is evidence, not proof

An attention heatmap can show that a generated placeholder position attends strongly to `PATIENT` or an unknown name position. That supports a hypothesis about model behaviour.

It does not prove causation. Test the hypothesis:

1. observe an attention pattern;
2. change or remove the suspected source token;
3. rerun the model;
4. compare output and attention;
5. repeat across examples.

A pretty heatmap alone is not an explanation.

## 14. Confidence is not correctness

The model always produces a distribution—even for unfamiliar prompts. A high top probability can result from a narrow, overfit model rather than genuine knowledge.

Calibration asks whether predictions made with, for example, 90% confidence are correct roughly 90% of the time. Tiny Mitra does not establish clinical calibration.

## 15. Safety boundary

This project uses synthetic, limited templates and a tiny word tokenizer. It demonstrates architecture and experimentation. It does not establish production readiness.

A real de-identification system would need:

- a validated PHI taxonomy;
- representative, governed datasets;
- strong access and privacy controls;
- direct leakage audits;
- adversarial and out-of-distribution tests;
- deterministic recognizers/rules around the generative model;
- human review and escalation;
- monitoring, versioning, and incident handling;
- legal, security, and clinical governance.

Never evaluate production safety using only synthetic exact match.

## 16. A useful per-example report

For every test case record:

```text
Input:
Expected:
Generated:
Entity condition: seen/unseen
Template condition: seen/unseen
Temperature:
Exact match:
Placeholder recall:
Placeholder precision:
Clinical preservation:
Raw identifier leaked:
Failure category:
Explanation:
```

This makes aggregate numbers auditable.

## 17. Questions to answer from results

1. Did lower validation loss reduce leakage?
2. Which identifier type was hardest?
3. Does performance fall more for unseen values or unseen templates?
4. Did extra blocks help preservation as well as redaction?
5. Did more heads learn distinct patterns?
6. At what temperature did output first become unstable?
7. Are failures consistent across seeds?
8. Does the model copy medical facts or genuinely use context?
9. Which failure would be most dangerous in practice?

## 18. Common misconceptions

**“95% token accuracy means 95% safe.”**  
No. Error severity is highly unequal.

**“Placeholder recall proves no PHI leaked.”**  
No. The placeholder and raw value can coexist.

**“Unseen names prove general language understanding.”**  
No. Familiar templates may identify their location.

**“Attention heatmaps explain the model.”**  
They are diagnostic evidence requiring counterfactual tests.

**“Synthetic success means production readiness.”**  
It only demonstrates success inside the synthetic experiment.

## Key takeaway

Tiny Mitra succeeds only when it removes identifiers and preserves clinical meaning. Evaluation must expose both sides, separate memorization from multiple forms of generalization, and report concrete failure cases rather than relying on one average metric.
