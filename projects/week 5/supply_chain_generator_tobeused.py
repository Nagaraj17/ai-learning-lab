"""
supply_chain_generator.py
=========================
Synthetic Supply Chain / GPO Workflow Dataset Generator
Designed for training the Tiny Transformer on a non-healthcare vocabulary.

Workflows generated:
1. Restock Flow: RECEIVE -> RESTOCK -> INVENTORY -> FORECAST -> ORDER
2. Fulfillment Flow: ORDER -> SHIPMENT -> RECEIVE -> RESTOCK
3. Contracting Flow: INVENTORY -> FORECAST -> SCENARIO -> CONTRACT
4. Finance Flow: PO -> SHIPMENT -> INVOICE -> RECONCILE
5. Vendor Flow: CONTRACT -> PURCHASE -> REBATE -> NCR

Curveballs (Noise):
- Delayed Shipments -> Re-Order
- Quality Check Failures
"""

import numpy as np
import random
from collections import Counter

# ---------------------------------------------------------------------------
# VOCABULARY: Supply Chain / GPO tokens
# ---------------------------------------------------------------------------
VOCAB = {
    "<PAD>":                  0,
    "<CASE_START>":           1,
    "RECEIVE":                2,
    "RESTOCK":                3,
    "INVENTORY":              4,
    "FORECAST":               5,
    "ORDER":                  6,
    "SHIPMENT":               7,
    "SCENARIO":               8,
    "CONTRACT":               9,
    "PO":                     10,
    "INVOICE":                11,
    "RECONCILE":              12,
    "PURCHASE":               13,
    "REBATE":                 14,
    "NCR":                    15,
    "SHIPMENT_DELAYED":       16,  # Noise/Curveball
    "SHIPMENT_LOST":          17,  # Noise/Curveball
    "RE_ORDER":               18,  # Noise/Curveball
    "QUALITY_CHECK_FAILED":   19,  # Noise/Curveball
    "AUDIT":                  20,  # Noise/Curveball
    "<CASE_END>":             21
}

ID2TOKEN = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)
PAD_ID = VOCAB["<PAD>"]

SCENARIO_FAMILIES = ["RESTOCK_FLOW", "FULFILLMENT_FLOW", "CONTRACTING_FLOW", "FINANCE_FLOW", "VENDOR_FLOW"]
VAL_ONLY_FAMILIES = set()
TEST_ONLY_FAMILIES = set()

# ---------------------------------------------------------------------------
# GENERATOR LOGIC
# ---------------------------------------------------------------------------

def _generate_single_case():
    """Generates a single synthetic supply chain case sequence."""
    seq = ["<CASE_START>"]
    
    # Randomly pick a core flow
    flow_type = random.choice([
        "RESTOCK_FLOW",
        "FULFILLMENT_FLOW",
        "CONTRACTING_FLOW",
        "FINANCE_FLOW",
        "VENDOR_FLOW"
    ])
    
    def add(token):
        # 15% chance to inject an random audit request mid-flow (Stochastic Noise)
        if random.random() < 0.15 and token not in ["<CASE_START>", "<CASE_END>"]:
            seq.append("AUDIT")
        seq.append(token)

    if flow_type == "RESTOCK_FLOW":
        add("RECEIVE")
        if random.random() < 0.1: # Curveball
            add("QUALITY_CHECK_FAILED")
            add("NCR")
        else:
            add("RESTOCK")
            add("INVENTORY")
            add("FORECAST")
            add("ORDER")
            
    elif flow_type == "FULFILLMENT_FLOW":
        add("ORDER")
        if random.random() < 0.2: # Curveball
            add("SHIPMENT_DELAYED")
            
        if random.random() < 0.05: # Severe Curveball
            add("SHIPMENT_LOST")
            add("RE_ORDER")
            add("SHIPMENT")
        else:
            add("SHIPMENT")
            
        add("RECEIVE")
        add("RESTOCK")
        
    elif flow_type == "CONTRACTING_FLOW":
        add("INVENTORY")
        add("FORECAST")
        add("SCENARIO")
        add("CONTRACT")
        
    elif flow_type == "FINANCE_FLOW":
        add("PO")
        add("SHIPMENT")
        add("INVOICE")
        if random.random() < 0.1: # Minor curveball
            add("AUDIT")
        add("RECONCILE")
        
    elif flow_type == "VENDOR_FLOW":
        add("CONTRACT")
        add("PURCHASE")
        add("REBATE")
        add("NCR")

    seq.append("<CASE_END>")
    
    # Convert string tokens to numerical IDs
    token_ids = [VOCAB[t] for t in seq]
    
    return {
        "flow_type": flow_type,
        "token_seq": token_ids,
        "text_seq": seq
    }

def generate_supply_chain_cases(num_cases=1200):
    """
    Generates a dataset of synthetic cases, removes duplicates, 
    and splits into train, val, and test sets.
    """
    all_cases = []
    seen_signatures = set()
    
    for _ in range(num_cases):
        case = _generate_single_case()
        sig = tuple(case["token_seq"])
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            all_cases.append(case)
            
    # Shuffle for randomness
    random.shuffle(all_cases)
    
    total = len(all_cases)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)
    
    train_cases = all_cases[:train_end]
    val_cases = all_cases[train_end:val_end]
    test_cases = all_cases[val_end:]
    
    stats = {
        "total_unique": total,
        "train": len(train_cases),
        "val": len(val_cases),
        "test": len(test_cases)
    }
    
    return train_cases, val_cases, test_cases, stats

def create_next_token_batches(cases, batch_size=32):
    """
    Converts a list of cases into batched X (input) and Y (target) tensors.
    Uses padding to ensure sequences in a batch have the same length.
    """
    batches = []
    # Shuffle for training robustness
    random.shuffle(cases)
    
    for i in range(0, len(cases), batch_size):
        batch_cases = cases[i : i + batch_size]
        
        # Find max sequence length in THIS batch
        max_len = max(len(c["token_seq"]) for c in batch_cases)
        
        X_batch = []
        Y_batch = []
        
        for c in batch_cases:
            seq = c["token_seq"]
            # To predict next token, X is seq[:-1], Y is seq[1:]
            x_seq = seq[:-1]
            y_seq = seq[1:]
            
            # Pad to (max_len - 1)
            pad_len = (max_len - 1) - len(x_seq)
            x_padded = x_seq + [PAD_ID] * pad_len
            y_padded = y_seq + [PAD_ID] * pad_len
            
            X_batch.append(x_padded)
            Y_batch.append(y_padded)
            
        batches.append((np.array(X_batch), np.array(Y_batch)))
        
    return batches

def validate_dataset(train, val, test):
    """Placeholder to match experiment_runner expectations"""
    return True

if __name__ == "__main__":
    train, val, test, stats = generate_supply_chain_cases(1000)
    print("Generation complete.")
    print(stats)
    print("Example Case:")
    print(" -> ".join(train[0]["text_seq"]))
