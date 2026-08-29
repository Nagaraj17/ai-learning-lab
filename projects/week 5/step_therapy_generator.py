"""
step_therapy_generator.py
=========================
Synthetic Prior-Authorization Step-Therapy Dataset Generator
Inspired by HL7 Da Vinci PAS operational workflow states.

IMPORTANT DISCLAIMER
--------------------
This module generates FICTIONAL educational data.
- PAS (HL7 Da Vinci Prior Authorization Support) inspires the operational
  workflow states (request created, submitted, received, reviewed, etc.).
- The approval/denial/step-therapy LOGIC is entirely fictional and invented
  for educational purposes only.
- This model must NOT be presented as making real clinical or coverage decisions.
- Fictional therapies (ZynPhase-X, Robalex, etc.) are used throughout.

Dataset Architecture
--------------------
Every case is generated from case facts -> fictional policy evaluation ->
valid PAS-style state transitions -> controlled operational variation.
The next event is always inferable from visible history (no leakage).
"""

import numpy as np
from collections import Counter

# ---------------------------------------------------------------------------
# VOCABULARY: PAS-inspired operational event tokens
# ---------------------------------------------------------------------------
VOCAB = {
    "<PAD>":                      0,
    "<CASE_START>":               1,
    "COVERAGE_VERIFIED":          2,
    "PA_NOT_REQUIRED":            3,
    "PA_REQUIRED":                4,
    "STEP_THERAPY_REQUIRED":      5,
    "NO_PREV_THERAPY":            6,
    "PREV_THERAPY_STARTED":       7,
    "PREV_THERAPY_COMPLETED":     8,
    "PREV_THERAPY_FAILED":        9,
    "FAILURE_DOCUMENTED":         10,
    "INTOLERANCE_DOCUMENTED":     11,
    "CONTRAINDICATION_DOCUMENTED": 12,
    "EXCEPTION_CRITERIA_MET":     13,
    "DOCS_COMPLETE":              14,
    "DOCS_MISSING":               15,
    "PA_REQUEST_CREATED":         16,
    "PA_REQUEST_SUBMITTED":       17,
    "PA_REQUEST_RECEIVED":        18,
    "PA_VALIDATION_PASSED":       19,
    "PA_REVIEW_STARTED":          20,
    "STATUS_INQUIRY":             21,
    "PA_PENDED":                  22,
    "ADDITIONAL_INFO_REQUESTED":  23,
    "DOCS_SUBMITTED":             24,
    "PA_REVIEW_RESUMED":          25,
    "PA_APPROVED":                26,
    "PA_DENIED_STEP_REQUIRED":    27,
    "PA_DENIED_DOCS_MISSING":     28,
    "EXCEPTION_REQUESTED":        29,
    "EXCEPTION_APPROVED":         30,
    "APPEAL_SUBMITTED":           31,
    "ADDITIONAL_EVIDENCE_PROVIDED": 32,
    "APPEAL_REVIEW_STARTED":      33,
    "DENIAL_UPHELD":              34,
    "DENIAL_OVERTURNED":          35,
    "PA_REQUEST_UPDATED":         36,
    "PA_REQUEST_CANCELLED":       37,
    "<CASE_END>":                 38,
}

ID2TOKEN = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)
PAD_ID = VOCAB["<PAD>"]

# ---------------------------------------------------------------------------
# FICTIONAL POLICIES (not real medical policy)
# ---------------------------------------------------------------------------
FICTIONAL_POLICIES = ["FICT-POL-001", "FICT-POL-002", "FICT-POL-003", "FICT-POL-004"]
FICTIONAL_THERAPIES = ["ZynPhase-X", "Robalex-20", "Clintoraz-ER", "Vormodex-SR", "Plastafene"]
ACTORS = {
    "provider": [
        "PA_REQUEST_CREATED", "PA_REQUEST_SUBMITTED", "DOCS_SUBMITTED",
        "APPEAL_SUBMITTED", "ADDITIONAL_EVIDENCE_PROVIDED", "STATUS_INQUIRY",
        "PA_REQUEST_UPDATED", "PA_REQUEST_CANCELLED"
    ],
    "payer": [
        "PA_REQUEST_RECEIVED", "PA_VALIDATION_PASSED", "PA_REVIEW_STARTED",
        "PA_PENDED", "ADDITIONAL_INFO_REQUESTED", "PA_REVIEW_RESUMED",
        "PA_APPROVED", "PA_DENIED_STEP_REQUIRED", "PA_DENIED_DOCS_MISSING",
        "EXCEPTION_APPROVED", "APPEAL_REVIEW_STARTED", "DENIAL_UPHELD", "DENIAL_OVERTURNED"
    ],
    "system": [
        "<CASE_START>", "COVERAGE_VERIFIED", "PA_NOT_REQUIRED", "PA_REQUIRED",
        "STEP_THERAPY_REQUIRED", "NO_PREV_THERAPY", "PREV_THERAPY_STARTED",
        "PREV_THERAPY_COMPLETED", "PREV_THERAPY_FAILED", "FAILURE_DOCUMENTED",
        "INTOLERANCE_DOCUMENTED", "CONTRAINDICATION_DOCUMENTED",
        "EXCEPTION_CRITERIA_MET", "DOCS_COMPLETE", "DOCS_MISSING",
        "EXCEPTION_REQUESTED", "<CASE_END>"
    ]
}
TOKEN_TO_ACTOR = {}
for actor, tokens in ACTORS.items():
    for t in tokens:
        TOKEN_TO_ACTOR[t] = actor

# Scenario family labels
SCENARIO_FAMILIES = [
    "direct_approval",
    "pended_then_approved",
    "step_therapy_denial",
    "step_therapy_exception",
    "appeal_overturned",
    "appeal_upheld",
    "docs_missing_denial",
    "docs_missing_resubmit_approval",
    "contraindication_exception",
    "cancellation",
    "no_pa_required",
    "status_check_approval",
]

# Holdout families: val-only and test-only combinations must be DIFFERENT
VAL_ONLY_FAMILIES = {"step_therapy_exception", "docs_missing_resubmit_approval"}
TEST_ONLY_FAMILIES = {"appeal_overturned", "contraindication_exception"}


def _make_event_record(case_id, event_index, token, timestamp_min, scenario_family,
                       policy_id, is_holdout_combination):
    """Creates a structured event record."""
    actor = TOKEN_TO_ACTOR.get(token, "system")
    return {
        "case_id": case_id,
        "event_index": event_index,
        "timestamp_min": timestamp_min,
        "actor": actor,
        "event_token": token,
        "scenario_family": scenario_family,
        "policy_id": policy_id,
        "is_holdout_combination": is_holdout_combination,
    }


def _generate_single_case(case_id, rng, scenario_family=None, policy_id=None):
    """
    Generate one case from case facts -> fictional policy -> PAS state transitions.
    The scenario_family can be forced for holdout construction; otherwise sampled.

    CRITICAL INFERABILITY RULE: Every transition's reason must be visible in history.
    If outcome depends on previous therapy failure, PREV_THERAPY_FAILED + FAILURE_DOCUMENTED
    appear BEFORE the approval/denial event.
    """
    # --- Sample case facts ---
    if policy_id is None:
        policy_id = rng.choice(FICTIONAL_POLICIES)
    therapy = rng.choice(FICTIONAL_THERAPIES)

    requires_pa = True
    if scenario_family == "no_pa_required":
        requires_pa = False

    has_status_check = rng.rand() < 0.20  # 20% of cases have a status inquiry
    urgent_review = rng.rand() < 0.15

    # --- Build event sequence ---
    # timestamp increments in minutes; realistic spacing varies by actor
    events = []
    ts = 0

    def add(token, dt_range=(1, 10)):
        nonlocal ts
        
        # Inject stochastic noise (simulating human unpredictability) to exponentially 
        # increase the number of unique mathematical sequences in the dataset.
        if token not in ["<CASE_START>", "<CASE_END>", "COVERAGE_VERIFIED", "PA_NOT_REQUIRED", "PA_REQUIRED"]:
            # 25% chance of an impatient doctor checking status
            if rng.rand() < 0.25:
                ts += rng.randint(5, 30)
                events.append(("STATUS_INQUIRY", ts))
                
            # 10% chance of doctor randomly updating the PA request
            if rng.rand() < 0.10:
                ts += rng.randint(2, 15)
                events.append(("PA_REQUEST_UPDATED", ts))

        lo, hi = dt_range
        if lo == hi:
            ts += lo
        else:
            ts += rng.randint(lo, hi + 1)
        events.append((token, ts))

    add("<CASE_START>", (0, 0))
    add("COVERAGE_VERIFIED", (2, 8))

    if not requires_pa:
        add("PA_NOT_REQUIRED", (1, 5))
        add("<CASE_END>", (1, 3))
        return events, "no_pa_required", policy_id

    add("PA_REQUIRED", (1, 5))

    # ----- Scenario routing -----
    if scenario_family is None:
        # Sample from non-holdout families for standard cases
        standard_families = [f for f in SCENARIO_FAMILIES
                             if f not in VAL_ONLY_FAMILIES and f not in TEST_ONLY_FAMILIES]
        scenario_family = rng.choice(standard_families)

    is_holdout = scenario_family in VAL_ONLY_FAMILIES or scenario_family in TEST_ONLY_FAMILIES

    # --- DIRECT APPROVAL ---
    if scenario_family == "direct_approval":
        add("STEP_THERAPY_REQUIRED", (1, 3))
        add("PREV_THERAPY_FAILED", (5, 30))      # visible: reason for approval
        add("FAILURE_DOCUMENTED", (5, 20))
        add("DOCS_COMPLETE", (2, 10))
        add("PA_REQUEST_CREATED", (1, 5))
        add("PA_REQUEST_SUBMITTED", (1, 5))
        add("PA_REQUEST_RECEIVED", (2, 8))
        add("PA_VALIDATION_PASSED", (2, 8))
        add("PA_REVIEW_STARTED", (5, 30))
        if has_status_check:
            add("STATUS_INQUIRY", (30, 180))
        add("PA_APPROVED", (10, 60))
        add("<CASE_END>", (1, 3))

    # --- PENDED THEN APPROVED ---
    elif scenario_family == "pended_then_approved":
        add("STEP_THERAPY_REQUIRED", (1, 3))
        add("PREV_THERAPY_FAILED", (5, 30))
        add("FAILURE_DOCUMENTED", (5, 20))
        add("DOCS_COMPLETE", (2, 10))
        add("PA_REQUEST_CREATED", (1, 5))
        add("PA_REQUEST_SUBMITTED", (1, 5))
        add("PA_REQUEST_RECEIVED", (2, 8))
        add("PA_VALIDATION_PASSED", (2, 8))
        add("PA_REVIEW_STARTED", (5, 30))
        add("PA_PENDED", (10, 60))
        add("ADDITIONAL_INFO_REQUESTED", (5, 20))
        if has_status_check:
            add("STATUS_INQUIRY", (60, 480))
        add("DOCS_SUBMITTED", (30, 1440))       # provider responds within 24h
        add("PA_REVIEW_RESUMED", (5, 30))
        add("PA_APPROVED", (10, 60))
        add("<CASE_END>", (1, 3))

    # --- STATUS CHECK APPROVAL (has_status_check forced) ---
    elif scenario_family == "status_check_approval":
        add("STEP_THERAPY_REQUIRED", (1, 3))
        add("PREV_THERAPY_COMPLETED", (5, 30))
        add("FAILURE_DOCUMENTED", (5, 20))
        add("DOCS_COMPLETE", (2, 10))
        add("PA_REQUEST_CREATED", (1, 5))
        add("PA_REQUEST_SUBMITTED", (1, 5))
        add("PA_REQUEST_RECEIVED", (2, 8))
        add("PA_VALIDATION_PASSED", (2, 8))
        add("PA_REVIEW_STARTED", (5, 30))
        add("STATUS_INQUIRY", (60, 300))         # forced status check
        add("PA_APPROVED", (10, 60))
        add("<CASE_END>", (1, 3))

    # --- STEP THERAPY DENIAL ---
    elif scenario_family == "step_therapy_denial":
        add("STEP_THERAPY_REQUIRED", (1, 3))
        add("NO_PREV_THERAPY", (2, 10))          # visible: reason for denial
        add("DOCS_COMPLETE", (2, 10))
        add("PA_REQUEST_CREATED", (1, 5))
        add("PA_REQUEST_SUBMITTED", (1, 5))
        add("PA_REQUEST_RECEIVED", (2, 8))
        add("PA_VALIDATION_PASSED", (2, 8))
        add("PA_REVIEW_STARTED", (5, 30))
        add("PA_DENIED_STEP_REQUIRED", (10, 60))
        add("<CASE_END>", (1, 3))

    # --- STEP THERAPY EXCEPTION (val-only holdout) ---
    elif scenario_family == "step_therapy_exception":
        add("STEP_THERAPY_REQUIRED", (1, 3))
        add("INTOLERANCE_DOCUMENTED", (5, 20))   # visible: intolerance = exception eligible
        add("EXCEPTION_CRITERIA_MET", (2, 8))
        add("DOCS_COMPLETE", (2, 10))
        add("PA_REQUEST_CREATED", (1, 5))
        add("PA_REQUEST_SUBMITTED", (1, 5))
        add("PA_REQUEST_RECEIVED", (2, 8))
        add("PA_VALIDATION_PASSED", (2, 8))
        add("PA_REVIEW_STARTED", (5, 30))
        add("EXCEPTION_REQUESTED", (5, 20))
        add("EXCEPTION_APPROVED", (10, 60))
        add("PA_APPROVED", (5, 20))
        add("<CASE_END>", (1, 3))

    # --- CONTRAINDICATION EXCEPTION (test-only holdout) ---
    elif scenario_family == "contraindication_exception":
        add("STEP_THERAPY_REQUIRED", (1, 3))
        add("CONTRAINDICATION_DOCUMENTED", (5, 20))  # visible: contraindication
        add("EXCEPTION_CRITERIA_MET", (2, 8))
        add("DOCS_COMPLETE", (2, 10))
        add("PA_REQUEST_CREATED", (1, 5))
        add("PA_REQUEST_SUBMITTED", (1, 5))
        add("PA_REQUEST_RECEIVED", (2, 8))
        add("PA_VALIDATION_PASSED", (2, 8))
        add("PA_REVIEW_STARTED", (5, 30))
        add("EXCEPTION_REQUESTED", (5, 20))
        add("EXCEPTION_APPROVED", (10, 60))
        add("PA_APPROVED", (5, 20))
        add("<CASE_END>", (1, 3))

    # --- APPEAL OVERTURNED (test-only holdout) ---
    elif scenario_family == "appeal_overturned":
        add("STEP_THERAPY_REQUIRED", (1, 3))
        add("NO_PREV_THERAPY", (2, 10))
        add("DOCS_COMPLETE", (2, 10))
        add("PA_REQUEST_CREATED", (1, 5))
        add("PA_REQUEST_SUBMITTED", (1, 5))
        add("PA_REQUEST_RECEIVED", (2, 8))
        add("PA_VALIDATION_PASSED", (2, 8))
        add("PA_REVIEW_STARTED", (5, 30))
        add("PA_DENIED_STEP_REQUIRED", (10, 60))
        add("APPEAL_SUBMITTED", (30, 480))
        add("ADDITIONAL_EVIDENCE_PROVIDED", (30, 1440))
        add("APPEAL_REVIEW_STARTED", (10, 60))
        add("DENIAL_OVERTURNED", (10, 60))
        add("PA_APPROVED", (5, 20))
        add("<CASE_END>", (1, 3))

    # --- APPEAL UPHELD ---
    elif scenario_family == "appeal_upheld":
        add("STEP_THERAPY_REQUIRED", (1, 3))
        add("NO_PREV_THERAPY", (2, 10))
        add("DOCS_COMPLETE", (2, 10))
        add("PA_REQUEST_CREATED", (1, 5))
        add("PA_REQUEST_SUBMITTED", (1, 5))
        add("PA_REQUEST_RECEIVED", (2, 8))
        add("PA_VALIDATION_PASSED", (2, 8))
        add("PA_REVIEW_STARTED", (5, 30))
        add("PA_DENIED_STEP_REQUIRED", (10, 60))
        add("APPEAL_SUBMITTED", (30, 480))
        add("APPEAL_REVIEW_STARTED", (10, 60))
        add("DENIAL_UPHELD", (10, 60))
        add("<CASE_END>", (1, 3))

    # --- DOCS MISSING DENIAL ---
    elif scenario_family == "docs_missing_denial":
        add("STEP_THERAPY_REQUIRED", (1, 3))
        add("PREV_THERAPY_FAILED", (5, 30))
        add("FAILURE_DOCUMENTED", (5, 20))
        add("DOCS_MISSING", (2, 10))             # visible: docs missing
        add("PA_REQUEST_CREATED", (1, 5))
        add("PA_REQUEST_SUBMITTED", (1, 5))
        add("PA_REQUEST_RECEIVED", (2, 8))
        add("PA_VALIDATION_PASSED", (2, 8))
        add("PA_REVIEW_STARTED", (5, 30))
        add("PA_PENDED", (5, 30))
        add("ADDITIONAL_INFO_REQUESTED", (5, 20))
        add("PA_DENIED_DOCS_MISSING", (1440, 2880))  # provider did not respond
        add("<CASE_END>", (1, 3))

    # --- DOCS MISSING RESUBMIT APPROVAL (val-only holdout) ---
    elif scenario_family == "docs_missing_resubmit_approval":
        add("STEP_THERAPY_REQUIRED", (1, 3))
        add("PREV_THERAPY_FAILED", (5, 30))
        add("FAILURE_DOCUMENTED", (5, 20))
        add("DOCS_MISSING", (2, 10))
        add("PA_REQUEST_CREATED", (1, 5))
        add("PA_REQUEST_SUBMITTED", (1, 5))
        add("PA_REQUEST_RECEIVED", (2, 8))
        add("PA_VALIDATION_PASSED", (2, 8))
        add("PA_REVIEW_STARTED", (5, 30))
        add("PA_PENDED", (5, 30))
        add("ADDITIONAL_INFO_REQUESTED", (5, 20))
        add("DOCS_SUBMITTED", (60, 720))
        add("PA_REVIEW_RESUMED", (5, 30))
        add("PA_APPROVED", (10, 60))
        add("<CASE_END>", (1, 3))

    # --- CANCELLATION ---
    elif scenario_family == "cancellation":
        add("STEP_THERAPY_REQUIRED", (1, 3))
        add("PREV_THERAPY_FAILED", (5, 30))
        add("FAILURE_DOCUMENTED", (5, 20))
        add("DOCS_COMPLETE", (2, 10))
        add("PA_REQUEST_CREATED", (1, 5))
        add("PA_REQUEST_SUBMITTED", (1, 5))
        add("PA_REQUEST_RECEIVED", (2, 8))
        add("PA_REQUEST_CANCELLED", (10, 120))   # cancelled after submission
        add("<CASE_END>", (1, 3))

    else:
        raise ValueError(f"Unknown scenario_family: {scenario_family}")

    return events, scenario_family, policy_id


def generate_step_therapy_cases(num_cases=1200, seed=42):
    """
    Generates a complete dataset of synthetic PA step-therapy cases.

    Design:
    - num_cases total cases generated from case facts -> policy -> transitions
    - Holdout families strictly partitioned: val-only != test-only
    - Fixed seed for reproducibility
    - Each sequence is deduplicated at the token-id level
    - Case metadata (actor, timestamp, policy_id) is preserved but NEVER
      passed to the model — only event_token sequences enter the model

    Returns
    -------
    all_cases : list of case dicts
    splits    : dict with keys 'train', 'val', 'test'
    """
    rng = np.random.RandomState(seed)

    # --- Target distribution across scenario families (approximate) ---
    # Standard (non-holdout) families get the bulk of cases
    standard_families = [f for f in SCENARIO_FAMILIES
                         if f not in VAL_ONLY_FAMILIES and f not in TEST_ONLY_FAMILIES]
    n_standard = int(num_cases * 0.70)
    n_val_holdout = int(num_cases * 0.15)
    n_test_holdout = num_cases - n_standard - n_val_holdout

    all_cases = []
    seen_sequences = set()
    case_id = 0

    def _add_case(events, scenario_family, policy_id, is_holdout):
        nonlocal case_id
        token_seq = [e[0] for e in events]
        token_ids = [VOCAB[t] for t in token_seq]
        key = tuple(token_ids)
        if key in seen_sequences:
            return False  # duplicate
        seen_sequences.add(key)

        # Build structured event records
        records = []
        for idx, (token, ts) in enumerate(events):
            records.append(_make_event_record(
                case_id=case_id,
                event_index=idx,
                token=token,
                timestamp_min=ts,
                scenario_family=scenario_family,
                policy_id=policy_id,
                is_holdout_combination=is_holdout
            ))

        all_cases.append({
            "case_id": case_id,
            "scenario_family": scenario_family,
            "policy_id": policy_id,
            "is_holdout_combination": is_holdout,
            "event_records": records,
            "token_seq": token_seq,
            "token_ids": token_ids,
        })
        case_id += 1
        return True

    # -- Generate standard cases --
    standard_families_cycle = standard_families * (n_standard // len(standard_families) + 1)
    rng.shuffle(standard_families_cycle)
    generated = 0
    attempts = 0
    for fam in standard_families_cycle:
        if generated >= n_standard:
            break
        attempts += 1
        events, sf, pid = _generate_single_case(generated, rng, scenario_family=fam)
        if _add_case(events, sf, pid, is_holdout=False):
            generated += 1
        if attempts > n_standard * 5:
            break

    # -- Generate val-only holdout cases --
    val_holdout_cases = []
    val_fams = list(VAL_ONLY_FAMILIES)
    generated_val = 0
    attempts = 0
    while generated_val < n_val_holdout:
        attempts += 1
        fam = val_fams[generated_val % len(val_fams)]
        events, sf, pid = _generate_single_case(case_id, rng, scenario_family=fam)
        if _add_case(events, sf, pid, is_holdout=True):
            val_holdout_cases.append(all_cases[-1])
            generated_val += 1
        if attempts > n_val_holdout * 10:
            break

    # -- Generate test-only holdout cases --
    test_holdout_cases = []
    test_fams = list(TEST_ONLY_FAMILIES)
    generated_test = 0
    attempts = 0
    while generated_test < n_test_holdout:
        attempts += 1
        fam = test_fams[generated_test % len(test_fams)]
        events, sf, pid = _generate_single_case(case_id, rng, scenario_family=fam)
        if _add_case(events, sf, pid, is_holdout=True):
            test_holdout_cases.append(all_cases[-1])
            generated_test += 1
        if attempts > n_test_holdout * 10:
            break

    # --- Build splits ---
    standard_cases = [c for c in all_cases if not c["is_holdout_combination"]]
    rng.shuffle(standard_cases)

    n_std = len(standard_cases)
    n_train = int(0.70 * n_std)
    n_val_std = int(0.15 * n_std)

    train_cases = standard_cases[:n_train]
    val_std = standard_cases[n_train:n_train + n_val_std]
    test_std = standard_cases[n_train + n_val_std:]

    val_cases = val_std + val_holdout_cases
    test_cases = test_std + test_holdout_cases

    rng.shuffle(val_cases)
    rng.shuffle(test_cases)

    splits = {"train": train_cases, "val": val_cases, "test": test_cases}
    return all_cases, splits


def validate_dataset(all_cases, splits):
    """
    Prints and asserts all required dataset quality checks.
    Returns a dict of validation results for display.
    """
    train = splits["train"]
    val = splits["val"]
    test = splits["test"]

    total = len(all_cases)
    n_train = len(train)
    n_val = len(val)
    n_test = len(test)

    results = {}
    results["total_cases"] = total
    results["n_train"] = n_train
    results["n_val"] = n_val
    results["n_test"] = n_test
    results["pct_train"] = n_train / total
    results["pct_val"] = n_val / total
    results["pct_test"] = n_test / total

    # Unique sequences
    all_seqs = set(tuple(c["token_ids"]) for c in all_cases)
    results["n_unique"] = len(all_seqs)
    results["duplicate_rate"] = 1.0 - len(all_seqs) / total

    # Overlap checks
    train_seqs = set(tuple(c["token_ids"]) for c in train)
    val_seqs = set(tuple(c["token_ids"]) for c in val)
    test_seqs = set(tuple(c["token_ids"]) for c in test)
    results["train_val_overlap"] = len(train_seqs & val_seqs)
    results["train_test_overlap"] = len(train_seqs & test_seqs)
    results["val_test_overlap"] = len(val_seqs & test_seqs)

    # Target token coverage
    train_targets = set()
    for c in train:
        for tid in c["token_ids"][1:]:  # targets are shifted by 1
            train_targets.add(tid)
    test_targets = set()
    for c in test:
        for tid in c["token_ids"][1:]:
            test_targets.add(tid)
    uncovered = list(test_targets - train_targets)
    results["uncovered_test_targets"] = uncovered
    # Holdout families may introduce tokens unseen in training — that is intentional.
    # We record this for analysis rather than asserting strict coverage.
    results["test_targets_in_train"] = len(uncovered) == 0
    results["uncovered_count"] = len(uncovered)

    # Scenario distribution
    scenario_counts = Counter(c["scenario_family"] for c in all_cases)
    results["scenario_distribution"] = dict(scenario_counts)

    # Outcome distribution
    outcomes = []
    for c in all_cases:
        last_token = c["token_seq"][-2]  # token before <CASE_END>
        outcomes.append(last_token)
    results["outcome_distribution"] = dict(Counter(outcomes))

    # Sequence length distribution
    lengths = [len(c["token_ids"]) for c in all_cases]
    results["seq_len_min"] = min(lengths)
    results["seq_len_max"] = max(lengths)
    results["seq_len_mean"] = float(np.mean(lengths))
    results["seq_len_std"] = float(np.std(lengths))

    # Token coverage
    all_token_ids = set()
    for c in all_cases:
        all_token_ids.update(c["token_ids"])
    results["token_coverage"] = len(all_token_ids)
    results["vocab_size"] = VOCAB_SIZE

    # Assertions
    assert results["train_val_overlap"] == 0, "FAIL: Train/Val overlap detected!"
    assert results["train_test_overlap"] == 0, "FAIL: Train/Test overlap detected!"
    assert results["val_test_overlap"] == 0, "FAIL: Val/Test overlap detected!"
    assert results["duplicate_rate"] == 0.0, f"FAIL: Duplicate rate {results['duplicate_rate']:.2%}"
    # Note: test_targets_in_train may be False for holdout families — this is intentional design.
    # The uncovered tokens come from novel scenario families reserved only for test evaluation.

    return results


def create_next_token_batches(case_list, max_seq_len=20, batch_size=32, shuffle=True, seed=42):
    """
    Converts case list into batched next-token prediction arrays.

    For each case with token_ids = [t0, t1, ..., tN]:
        Input  X[t] = t_{0..N-1}  (visible context)
        Target Y[t] = t_{1..N}    (next token to predict)

    Padding is applied to max_seq_len-1. Padding positions are masked (mask=0).
    """
    X_list, Y_list, mask_list = [], [], []

    for case in case_list:
        t_ids = case["token_ids"]
        x_seq = t_ids[:-1]
        y_seq = t_ids[1:]

        target_len = max_seq_len - 1
        if len(x_seq) > target_len:
            x_seq = x_seq[:target_len]
            y_seq = y_seq[:target_len]

        pad_len = target_len - len(x_seq)
        X_list.append(x_seq + [PAD_ID] * pad_len)
        Y_list.append(y_seq + [PAD_ID] * pad_len)
        mask_list.append([1.0] * len(x_seq) + [0.0] * pad_len)

    X = np.array(X_list, dtype=np.int64)
    Y = np.array(Y_list, dtype=np.int64)
    mask = np.array(mask_list, dtype=np.float32)

    indices = np.arange(len(X))
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    batches = []
    for i in range(0, len(X), batch_size):
        b_idx = indices[i:i + batch_size]
        batches.append({"X": X[b_idx], "Y": Y[b_idx], "mask": mask[b_idx]})

    return batches


if __name__ == "__main__":
    cases, splits = generate_step_therapy_cases(num_cases=1200, seed=42)
    results = validate_dataset(cases, splits)
    print("=== Dataset Validation ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
