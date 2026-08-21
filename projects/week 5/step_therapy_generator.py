import numpy as np

# Vocabulary Definition
VOCAB = {
    "<PAD>": 0,
    "SUBMIT_PA": 1,
    "DOC_COMPLETE": 2,
    "DOC_MISSING": 3,
    "PREV_THERAPY_FAIL": 4,
    "PREV_THERAPY_NONE": 5,
    "PEND_INFO": 6,
    "ADDITIONAL_EVIDENCE": 7,
    "CONTRAINDICATION": 8,
    "CLINICAL_EXCEPTION": 9,
    "APPEAL_SUBMIT": 10,
    "OVERTURN": 11,
    "UPHELD": 12,
    "DIRECT_APPROVE": 13,
    "DENIAL_STEP_THERAPY": 14,
    "RESUBMIT": 15,
    "CANCEL": 16,
}

ID2TOKEN = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)
PAD_ID = VOCAB["<PAD>"]

# Branch templates
BRANCH_TEMPLATES = {
    "direct_approve": [
        "SUBMIT_PA", "DOC_COMPLETE", "PREV_THERAPY_FAIL", "DIRECT_APPROVE"
    ],
    "pended_approve": [
        "SUBMIT_PA", "DOC_COMPLETE", "PREV_THERAPY_FAIL", "PEND_INFO", "ADDITIONAL_EVIDENCE", "DIRECT_APPROVE"
    ],
    "missing_therapy_denial": [
        "SUBMIT_PA", "DOC_COMPLETE", "PREV_THERAPY_NONE", "DENIAL_STEP_THERAPY"
    ],
    "missing_doc_pended_approve": [
        "SUBMIT_PA", "DOC_MISSING", "PEND_INFO", "ADDITIONAL_EVIDENCE", "PREV_THERAPY_FAIL", "DIRECT_APPROVE"
    ],
    "missing_doc_pended_denial": [
        "SUBMIT_PA", "DOC_MISSING", "PEND_INFO", "ADDITIONAL_EVIDENCE", "PREV_THERAPY_NONE", "DENIAL_STEP_THERAPY"
    ],
    "contraindication_exception": [
        "SUBMIT_PA", "CONTRAINDICATION", "CLINICAL_EXCEPTION", "DIRECT_APPROVE"
    ],
    "appeal_overturned": [
        "SUBMIT_PA", "DOC_COMPLETE", "PREV_THERAPY_NONE", "DENIAL_STEP_THERAPY", "APPEAL_SUBMIT", "ADDITIONAL_EVIDENCE", "OVERTURN"
    ],
    "appeal_upheld": [
        "SUBMIT_PA", "DOC_COMPLETE", "PREV_THERAPY_NONE", "DENIAL_STEP_THERAPY", "APPEAL_SUBMIT", "UPHELD"
    ],
    "cancellation": [
        "SUBMIT_PA", "DOC_MISSING", "CANCEL"
    ],
    "resubmission_approve": [
        "SUBMIT_PA", "DOC_MISSING", "RESUBMIT", "DOC_COMPLETE", "PREV_THERAPY_FAIL", "DIRECT_APPROVE"
    ],
}

# Combinatorial holdouts reserved strictly for Val / Test sets
COMBINATORIAL_HOLDOUT_BRANCHES = {"missing_doc_pended_approve", "appeal_overturned"}

def generate_step_therapy_cases(num_cases=1000, seed=42, max_len=10):
    """
    Generates synthetic step-therapy prior-authorization cases.
    Returns:
        cases: list of dicts with 'case_id', 'tokens', 'token_ids', 'branch'
        splits: dict mapping 'train', 'val', 'test' to lists of case dicts
    """
    rng = np.random.RandomState(seed)
    branch_names = list(BRANCH_TEMPLATES.keys())
    
    all_cases = []
    
    for i in range(num_cases):
        # Pick a branch template
        branch = rng.choice(branch_names)
        tokens = BRANCH_TEMPLATES[branch]
        token_ids = [VOCAB[t] for t in tokens]
        
        case_dict = {
            "case_id": i,
            "branch": branch,
            "tokens": tokens,
            "token_ids": token_ids,
            "is_holdout": branch in COMBINATORIAL_HOLDOUT_BRANCHES
        }
        all_cases.append(case_dict)

    # Separate holdout cases and standard cases
    standard_cases = [c for c in all_cases if not c["is_holdout"]]
    holdout_cases = [c for c in all_cases if c["is_holdout"]]

    # Shuffle standard cases by seed
    rng.shuffle(standard_cases)

    # Split standard cases: 70% train, 15% val, 15% test
    n_std = len(standard_cases)
    n_train = int(0.70 * n_std)
    n_val_std = int(0.15 * n_std)
    
    train_cases = standard_cases[:n_train]
    val_std = standard_cases[n_train:n_train + n_val_std]
    test_std = standard_cases[n_train + n_val_std:]

    # Distribute combinatorial holdouts 50/50 between val and test
    rng.shuffle(holdout_cases)
    n_hold = len(holdout_cases)
    val_hold = holdout_cases[:n_hold // 2]
    test_hold = holdout_cases[n_hold // 2:]

    val_cases = val_std + val_hold
    test_cases = test_std + test_hold

    rng.shuffle(val_cases)
    rng.shuffle(test_cases)

    splits = {
        "train": train_cases,
        "val": val_cases,
        "test": test_cases
    }

    return all_cases, splits

def create_dataset_batches(case_list, max_seq_len=10, batch_size=32, shuffle=True, seed=42):
    """
    Converts cases into input (X) and target (Y) arrays padded to max_seq_len.
    Task: Next-token prediction.
    X[t] -> Y[t] (predict sequence shifted by 1 token).
    """
    X_list = []
    Y_list = []
    mask_list = []

    for case in case_list:
        t_ids = case["token_ids"]
        # Sequence input: t_ids[:-1], target: t_ids[1:]
        x_seq = t_ids[:-1]
        y_seq = t_ids[1:]
        
        # Pad to max_seq_len - 1
        target_len = max_seq_len - 1
        pad_len = target_len - len(x_seq)
        
        x_padded = x_seq + [PAD_ID] * pad_len
        y_padded = y_seq + [PAD_ID] * pad_len
        valid_mask = [1] * len(x_seq) + [0] * pad_len

        X_list.append(x_padded)
        Y_list.append(y_padded)
        mask_list.append(valid_mask)

    X = np.array(X_list, dtype=np.int64)
    Y = np.array(Y_list, dtype=np.int64)
    mask = np.array(mask_list, dtype=np.float32)

    num_samples = len(X)
    indices = np.arange(num_samples)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    batches = []
    for i in range(0, num_samples, batch_size):
        batch_idx = indices[i:i + batch_size]
        batches.append({
            "X": X[batch_idx],
            "Y": Y[batch_idx],
            "mask": mask[batch_idx]
        })

    return batches

if __name__ == "__main__":
    cases, splits = generate_step_therapy_cases(num_cases=1000, seed=42)
    print(f"Total Cases: {len(cases)}")
    print(f"Train Cases: {len(splits['train'])}")
    print(f"Val Cases: {len(splits['val'])}")
    print(f"Test Cases: {len(splits['test'])}")
    print(f"Vocab size: {VOCAB_SIZE}")
    
    batches = create_dataset_batches(splits["train"], batch_size=32)
    print(f"Train Batches: {len(batches)}")
    print("Sample Batch X shape:", batches[0]["X"].shape)
    print("Sample Batch Y shape:", batches[0]["Y"].shape)
