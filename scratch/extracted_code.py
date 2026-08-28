
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json

VOCAB = {
    "<PAD>": 0,
    "<CASE_START>": 1,
    "COVERAGE_CHECKED": 2,
    "PA_NOT_REQUIRED": 3,
    "PA_REQUIRED": 4,
    "STEP_THERAPY_REQUIRED": 5,
    "PREVIOUS_THERAPY_STARTED": 6,
    "PREVIOUS_THERAPY_COMPLETED": 7,
    "PREVIOUS_THERAPY_FAILED": 8,
    "NO_PREVIOUS_THERAPY": 9,
    "FAILURE_DOCUMENTED": 10,
    "INTOLERANCE_DOCUMENTED": 11,
    "CONTRAINDICATION_DOCUMENTED": 12,
    "DOCUMENTATION_COMPLETE": 13,
    "DOCUMENTATION_MISSING": 14,
    "PA_REQUEST_CREATED": 15,
    "PA_REQUEST_SUBMITTED": 16,
    "PA_REQUEST_RECEIVED": 17,
    "PA_REVIEW_STARTED": 18,
    "PA_PENDED": 19,
    "ADDITIONAL_INFO_REQUESTED": 20,
    "DOCUMENTATION_SUBMITTED": 21,
    "PA_REVIEW_RESUMED": 22,
    "PA_APPROVED": 23,
    "PA_DENIED_STEP_REQUIRED": 24,
    "PA_DENIED_DOCUMENTATION_MISSING": 25,
    "EXCEPTION_REQUESTED": 26,
    "EXCEPTION_APPROVED": 27,
    "APPEAL_SUBMITTED": 28,
    "ADDITIONAL_EVIDENCE_SUBMITTED": 29,
    "APPEAL_REVIEW_STARTED": 30,
    "DENIAL_UPHELD": 31,
    "DENIAL_OVERTURNED": 32,
    "PA_REQUEST_UPDATED": 33,
    "PA_REQUEST_CANCELLED": 34,
    "CASE_CLOSED": 35
}

ID2TOKEN = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)
PAD_ID = VOCAB["<PAD>"]

def generate_case(rng):
    """Generates a single case using a state machine based on sampled facts."""
    # 1. Sample Facts
    requires_pa = rng.rand() < 0.8
    requires_step_therapy = requires_pa and rng.rand() < 0.7
    
    prev_status = rng.choice(["failed", "completed", "none", "intolerance", "contraindication"])
    docs_missing = rng.rand() < 0.3
    docs_resubmitted = docs_missing and rng.rand() < 0.8
    
    appeal_success = rng.rand() < 0.5
    exception_success = rng.rand() < 0.8
    is_cancelled = rng.rand() < 0.05
    
    # 2. Build Event Sequence
    seq = ["<CASE_START>", "COVERAGE_CHECKED"]
    
    if not requires_pa:
        seq.extend(["PA_NOT_REQUIRED", "CASE_CLOSED"])
        return seq, "pa_not_required"
        
    seq.append("PA_REQUIRED")
    
    if requires_step_therapy:
        seq.append("STEP_THERAPY_REQUIRED")
        if prev_status == "failed":
            seq.extend(["PREVIOUS_THERAPY_FAILED", "FAILURE_DOCUMENTED"])
        elif prev_status == "completed":
            seq.append("PREVIOUS_THERAPY_COMPLETED")
        elif prev_status == "intolerance":
            seq.append("INTOLERANCE_DOCUMENTED")
        elif prev_status == "contraindication":
            seq.append("CONTRAINDICATION_DOCUMENTED")
        else:
            seq.append("NO_PREVIOUS_THERAPY")
            
    seq.append("PA_REQUEST_CREATED")
    
    if docs_missing:
        seq.append("DOCUMENTATION_MISSING")
    else:
        seq.append("DOCUMENTATION_COMPLETE")
        
    if is_cancelled:
        seq.extend(["PA_REQUEST_CANCELLED", "CASE_CLOSED"])
        return seq, "cancelled"
        
    seq.extend(["PA_REQUEST_SUBMITTED", "PA_REQUEST_RECEIVED", "PA_REVIEW_STARTED"])
    
    if docs_missing:
        seq.extend(["PA_PENDED", "ADDITIONAL_INFO_REQUESTED"])
        if not docs_resubmitted:
            seq.extend(["PA_DENIED_DOCUMENTATION_MISSING", "CASE_CLOSED"])
            return seq, "denial_missing_doc"
        seq.extend(["DOCUMENTATION_SUBMITTED", "PA_REVIEW_RESUMED", "DOCUMENTATION_COMPLETE"])
        
    if requires_step_therapy and prev_status == "none":
        seq.append("PA_DENIED_STEP_REQUIRED")
        
        # Exception or Appeal path
        if rng.rand() < 0.3:
            seq.append("EXCEPTION_REQUESTED")
            if exception_success:
                seq.extend(["EXCEPTION_APPROVED", "PA_APPROVED", "CASE_CLOSED"])
                return seq, "exception_approved"
            else:
                seq.extend(["DENIAL_UPHELD", "CASE_CLOSED"])
                return seq, "exception_denied"
        else:
            seq.extend(["APPEAL_SUBMITTED", "ADDITIONAL_EVIDENCE_SUBMITTED", "APPEAL_REVIEW_STARTED"])
            if appeal_success:
                seq.extend(["DENIAL_OVERTURNED", "PA_APPROVED", "CASE_CLOSED"])
                return seq, "appeal_overturned"
            else:
                seq.extend(["DENIAL_UPHELD", "CASE_CLOSED"])
                return seq, "appeal_upheld"
                
    seq.extend(["PA_APPROVED", "CASE_CLOSED"])
    return seq, "standard_approval"

def generate_dataset(num_cases=3000, seed=42):
    rng = np.random.RandomState(seed)
    
    # Generate cases
    all_cases = []
    unique_seqs = set()
    
    while len(all_cases) < num_cases:
        seq, scenario = generate_case(rng)
        seq_tuple = tuple(seq)
        if seq_tuple not in unique_seqs:
            unique_seqs.add(seq_tuple)
            all_cases.append({
                "case_id": len(all_cases),
                "scenario": scenario,
                "tokens": seq,
                "token_ids": [VOCAB[t] for t in seq]
            })
            
    # Assign holdout logic based on scenario
    # We will hold out specific scenarios entirely for val/test
    # appeal_overturned -> Test only
    # exception_approved -> Val only
    # appeal_upheld -> Test only
    # exception_denied -> Val only
    
    train_cases, val_cases, test_cases = [], [], []
    
    for case in all_cases:
        if case["scenario"] in ["exception_approved", "exception_denied"]:
            val_cases.append(case)
        elif case["scenario"] in ["appeal_overturned", "appeal_upheld"]:
            test_cases.append(case)
        else:
            train_cases.append(case)
            
    # Shuffle
    rng.shuffle(train_cases)
    rng.shuffle(val_cases)
    rng.shuffle(test_cases)
    
    # We want a 70/15/15 split. To achieve this strictly, we will sample standard cases into val/test to fill quotas.
    n_total = len(all_cases)
    target_val = int(0.15 * n_total)
    target_test = int(0.15 * n_total)
    
    # Move some train cases to val/test if needed
    while len(val_cases) < target_val and len(train_cases) > 0:
        val_cases.append(train_cases.pop())
    while len(test_cases) < target_test and len(train_cases) > 0:
        test_cases.append(train_cases.pop())
        
    return {"train": train_cases, "val": val_cases, "test": test_cases}, all_cases

splits, all_cases = generate_dataset(num_cases=2000, seed=42)

def validate_dataset(splits):
    print("--- Dataset Validation ---")
    train_seqs = set(tuple(c['token_ids']) for c in splits['train'])
    val_seqs = set(tuple(c['token_ids']) for c in splits['val'])
    test_seqs = set(tuple(c['token_ids']) for c in splits['test'])
    
    assert len(train_seqs.intersection(val_seqs)) == 0, "Overlap between Train and Val!"
    assert len(train_seqs.intersection(test_seqs)) == 0, "Overlap between Train and Test!"
    assert len(val_seqs.intersection(test_seqs)) == 0, "Overlap between Val and Test!"
    print("[OK] Zero exact-sequence overlap.")
    
    total = len(splits['train']) + len(splits['val']) + len(splits['test'])
    print(f"Split sizes: Train={len(splits['train'])} ({len(splits['train'])/total:.1%}), Val={len(splits['val'])} ({len(splits['val'])/total:.1%}), Test={len(splits['test'])} ({len(splits['test'])/total:.1%})")
    
    all_ids = set()
    for c in all_cases:
        all_ids.update(c['token_ids'])
    print(f"[OK] Target-vocabulary coverage: {len(all_ids)} / {VOCAB_SIZE} tokens used.")
    
    lens = [len(c['token_ids']) for c in all_cases]
    print(f"[OK] Sequence lengths: min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.1f}")

validate_dataset(splits)

def create_dataset_batches(case_list, max_seq_len=24, batch_size=32, shuffle=True, seed=42):
    X_list, Y_list, mask_list = [], [], []
    for case in case_list:
        t_ids = case["token_ids"]
        x_seq, y_seq = t_ids[:-1], t_ids[1:]
        target_len = max_seq_len - 1
        pad_len = target_len - len(x_seq)
        
        # Truncate if too long
        if pad_len < 0:
            x_seq = x_seq[:target_len]
            y_seq = y_seq[:target_len]
            pad_len = 0
            
        X_list.append(x_seq + [PAD_ID] * pad_len)
        Y_list.append(y_seq + [PAD_ID] * pad_len)
        mask_list.append([1] * len(x_seq) + [0] * pad_len)

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




def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def sinusoidal_positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    position = np.arange(0, seq_len, dtype=np.float64)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float64) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    if d_model % 2 == 1:
        pe[:, 1::2] = np.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = np.cos(position * div_term)
    return pe

class LayerNormNumPy:
    def __init__(self, d_model, eps=1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model, dtype=np.float64)
        self.beta = np.zeros(d_model, dtype=np.float64)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self.cache = None

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)
        x_norm = (x - mean) / std
        out = self.gamma * x_norm + self.beta
        self.cache = (x, x_norm, mean, std)
        return out

    def backward(self, dout):
        x, x_norm, mean, std = self.cache
        self.dgamma = np.sum(dout * x_norm, axis=(0, 1))
        self.dbeta = np.sum(dout, axis=(0, 1))
        dx_norm = dout * self.gamma
        N = self.d_model
        dx = (1.0 / (N * std)) * (
            N * dx_norm 
            - np.sum(dx_norm, axis=-1, keepdims=True) 
            - x_norm * np.sum(dx_norm * x_norm, axis=-1, keepdims=True)
        )
        return dx

class FeedForwardNumPy:
    def __init__(self, d_model, d_ff, rng):
        self.d_model = d_model
        self.d_ff = d_ff
        limit1 = np.sqrt(6.0 / (d_model + d_ff))
        self.W1 = rng.uniform(-limit1, limit1, (d_model, d_ff)).astype(np.float64)
        self.b1 = np.zeros(d_ff, dtype=np.float64)
        limit2 = np.sqrt(6.0 / (d_ff + d_model))
        self.W2 = rng.uniform(-limit2, limit2, (d_ff, d_model)).astype(np.float64)
        self.b2 = np.zeros(d_model, dtype=np.float64)
        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)
        self.cache = None

    def forward(self, x):
        h1 = np.matmul(x, self.W1) + self.b1
        a1 = np.maximum(0, h1)
        out = np.matmul(a1, self.W2) + self.b2
        self.cache = (x, h1, a1)
        return out

    def backward(self, dout):
        x, h1, a1 = self.cache
        self.dW2 = np.matmul(a1.reshape(-1, self.d_ff).T, dout.reshape(-1, self.d_model))
        self.db2 = np.sum(dout, axis=(0, 1))
        da1 = np.matmul(dout, self.W2.T)
        dh1 = da1 * (h1 > 0)
        self.dW1 = np.matmul(x.reshape(-1, self.d_model).T, dh1.reshape(-1, self.d_ff))
        self.db1 = np.sum(dh1, axis=(0, 1))
        dx = np.matmul(dh1, self.W1.T)
        return dx

class MultiHeadAttentionNumPy:
    def __init__(self, d_model, num_heads, d_head, rng):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_head
        self.d_inner = num_heads * d_head
        limit = np.sqrt(6.0 / (d_model + self.d_inner))
        self.WQ = rng.uniform(-limit, limit, (d_model, self.d_inner)).astype(np.float64)
        self.WK = rng.uniform(-limit, limit, (d_model, self.d_inner)).astype(np.float64)
        self.WV = rng.uniform(-limit, limit, (d_model, self.d_inner)).astype(np.float64)
        self.WO = rng.uniform(-limit, limit, (self.d_inner, d_model)).astype(np.float64)
        self.bO = np.zeros(d_model, dtype=np.float64)
        self.dWQ = np.zeros_like(self.WQ)
        self.dWK = np.zeros_like(self.WK)
        self.dWV = np.zeros_like(self.WV)
        self.dWO = np.zeros_like(self.WO)
        self.dbO = np.zeros_like(self.bO)
        self.cache = None

    def forward(self, x):
        B, T, _ = x.shape
        h, d_k = self.num_heads, self.d_k
        Q = np.matmul(x, self.WQ).reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
        K = np.matmul(x, self.WK).reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
        V = np.matmul(x, self.WV).reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        causal_mask = np.triu(np.ones((T, T), dtype=np.float64), k=1) * -1e9
        scores = scores + causal_mask[np.newaxis, np.newaxis, :, :]
        attn_weights = softmax(scores, axis=-1)
        context = np.matmul(attn_weights, V)
        context_concat = context.transpose(0, 2, 1, 3).reshape(B, T, self.d_inner)
        out = np.matmul(context_concat, self.WO) + self.bO
        self.cache = (x, Q, K, V, attn_weights, context_concat)
        return out

    def backward(self, dout):
        x, Q, K, V, attn_weights, context_concat = self.cache
        B, T, _ = x.shape
        h, d_k = self.num_heads, self.d_k
        self.dWO = np.matmul(context_concat.reshape(-1, self.d_inner).T, dout.reshape(-1, self.d_model))
        self.dbO = np.sum(dout, axis=(0, 1))
        dcontext_concat = np.matmul(dout, self.WO.T)
        dcontext = dcontext_concat.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
        dV = np.matmul(attn_weights.transpose(0, 1, 3, 2), dcontext)
        dattn_weights = np.matmul(dcontext, V.transpose(0, 1, 3, 2))
        dscores = attn_weights * (dattn_weights - np.sum(dattn_weights * attn_weights, axis=-1, keepdims=True))
        dscores = dscores / np.sqrt(d_k)
        dQ = np.matmul(dscores, K)
        dK = np.matmul(dscores.transpose(0, 1, 3, 2), Q)
        dQ_flat = dQ.transpose(0, 2, 1, 3).reshape(B * T, self.d_inner)
        dK_flat = dK.transpose(0, 2, 1, 3).reshape(B * T, self.d_inner)
        dV_flat = dV.transpose(0, 2, 1, 3).reshape(B * T, self.d_inner)
        x_flat = x.reshape(B * T, self.d_model)
        self.dWQ = np.matmul(x_flat.T, dQ_flat)
        self.dWK = np.matmul(x_flat.T, dK_flat)
        self.dWV = np.matmul(x_flat.T, dV_flat)
        dx = np.matmul(dQ_flat, self.WQ.T) + np.matmul(dK_flat, self.WK.T) + np.matmul(dV_flat, self.WV.T)
        return dx.reshape(B, T, self.d_model)

class TransformerBlockNumPy:
    def __init__(self, d_model, num_heads, d_head, d_ff, rng, use_ffn=True, use_ln=True, use_res=True):
        self.use_ffn = use_ffn
        self.use_ln = use_ln
        self.use_res = use_res
        self.attn = MultiHeadAttentionNumPy(d_model, num_heads, d_head, rng)
        self.ln1 = LayerNormNumPy(d_model) if use_ln else None
        if use_ffn:
            self.ffn = FeedForwardNumPy(d_model, d_ff, rng)
            self.ln2 = LayerNormNumPy(d_model) if use_ln else None

    def forward(self, x):
        norm1 = self.ln1.forward(x) if self.use_ln else x
        attn_out = self.attn.forward(norm1)
        x1 = (x + attn_out) if self.use_res else attn_out
        
        if self.use_ffn:
            norm2 = self.ln2.forward(x1) if self.use_ln else x1
            ffn_out = self.ffn.forward(norm2)
            x2 = (x1 + ffn_out) if self.use_res else ffn_out
        else:
            x2 = x1
        return x2

    def backward(self, dout):
        if self.use_ffn:
            dffn_out = dout if self.use_res else dout
            dx1_from_ffn = self.ffn.backward(dffn_out)
            dnorm2 = self.ln2.backward(dx1_from_ffn) if self.use_ln else dx1_from_ffn
            dx1 = (dout + dnorm2) if self.use_res else dnorm2
        else:
            dx1 = dout
            
        dattn_out = dx1 if self.use_res else dx1
        dnorm1 = self.attn.backward(dattn_out)
        dx_from_attn = self.ln1.backward(dnorm1) if self.use_ln else dnorm1
        dx = (dx1 + dx_from_attn) if self.use_res else dx_from_attn
        return dx




def compute_cross_entropy_loss(logits, targets, mask):
    probs = softmax(logits, axis=-1)
    B, T, V = logits.shape
    batch_idx = np.arange(B)[:, np.newaxis]
    time_idx = np.arange(T)[np.newaxis, :]
    target_probs = probs[batch_idx, time_idx, targets]
    log_probs = np.log(np.maximum(target_probs, 1e-12))
    masked_loss = -log_probs * mask
    total_valid = np.sum(mask)
    loss = np.sum(masked_loss) / np.maximum(total_valid, 1.0)
    
    dlogits = probs.copy()
    dlogits[batch_idx, time_idx, targets] -= 1.0
    dlogits = dlogits * mask[:, :, np.newaxis] / np.maximum(total_valid, 1.0)
    return loss, dlogits, probs

class ModularTinyTransformer:
    def __init__(self, model_id, vocab_size, d_model=24, d_ff=96, max_len=24, seed=42):
        self.model_id = model_id
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.max_len = max_len
        rng = np.random.RandomState(seed)

        limit_emb = np.sqrt(6.0 / (vocab_size + d_model))
        self.W_emb = rng.uniform(-limit_emb, limit_emb, (vocab_size, d_model)).astype(np.float64)
        self.pos_enc = sinusoidal_positional_encoding(max_len, d_model)
        self.dW_emb = np.zeros_like(self.W_emb)

        self.blocks = []
        if model_id == "A":
            pass
        elif model_id == "B":
            self.blocks.append(MultiHeadAttentionNumPy(d_model, num_heads=1, d_head=24, rng=rng))
        elif model_id == "C":
            self.blocks.append(MultiHeadAttentionNumPy(d_model, num_heads=4, d_head=6, rng=rng))
        elif model_id == "D":
            self.blocks.append(TransformerBlockNumPy(d_model, 4, 6, d_ff, rng))
            self.blocks.append(TransformerBlockNumPy(d_model, 4, 6, d_ff, rng))
        elif model_id == "D-1":
            self.blocks.append(TransformerBlockNumPy(d_model, 4, 6, d_ff, rng))
        elif model_id == "D-no-FFN":
            self.blocks.append(TransformerBlockNumPy(d_model, 4, 6, d_ff, rng, use_ffn=False))
            self.blocks.append(TransformerBlockNumPy(d_model, 4, 6, d_ff, rng, use_ffn=False))
        elif model_id == "D-no-LN":
            self.blocks.append(TransformerBlockNumPy(d_model, 4, 6, d_ff, rng, use_ln=False))
            self.blocks.append(TransformerBlockNumPy(d_model, 4, 6, d_ff, rng, use_ln=False))
        elif model_id == "D-no-res":
            self.blocks.append(TransformerBlockNumPy(d_model, 4, 6, d_ff, rng, use_res=False))
            self.blocks.append(TransformerBlockNumPy(d_model, 4, 6, d_ff, rng, use_res=False))
        else:
            raise ValueError(f"Unknown model_id: {model_id}")

        limit_head = np.sqrt(6.0 / (d_model + vocab_size))
        self.W_head = rng.uniform(-limit_head, limit_head, (d_model, vocab_size)).astype(np.float64)
        self.b_head = np.zeros(vocab_size, dtype=np.float64)
        self.dW_head = np.zeros_like(self.W_head)
        self.db_head = np.zeros_like(self.b_head)
        self.input_ids = None

    def get_num_params(self):
        total = self.W_emb.size + self.W_head.size + self.b_head.size
        for block in self.blocks:
            if isinstance(block, MultiHeadAttentionNumPy):
                total += block.WQ.size + block.WK.size + block.WV.size + block.WO.size + block.bO.size
            elif isinstance(block, TransformerBlockNumPy):
                attn = block.attn
                total += attn.WQ.size + attn.WK.size + attn.WV.size + attn.WO.size + attn.bO.size
                if block.use_ln: total += block.ln1.gamma.size + block.ln1.beta.size
                if block.use_ffn:
                    total += block.ffn.W1.size + block.ffn.b1.size + block.ffn.W2.size + block.ffn.b2.size
                    if block.use_ln: total += block.ln2.gamma.size + block.ln2.beta.size
        return total

    def forward(self, input_ids):
        self.input_ids = input_ids
        B, T = input_ids.shape
        emb = self.W_emb[input_ids]
        x = emb + self.pos_enc[:T]
        for block in self.blocks:
            if hasattr(block, 'forward'):
                x = block.forward(x)
        logits = np.matmul(x, self.W_head) + self.b_head
        return logits, x

    def backward(self, dlogits, x_final):
        self.dW_head = np.matmul(x_final.reshape(-1, self.d_model).T, dlogits.reshape(-1, self.vocab_size))
        self.db_head = np.sum(dlogits, axis=(0, 1))
        dx = np.matmul(dlogits, self.W_head.T)
        for block in reversed(self.blocks):
            if hasattr(block, 'backward'):
                dx = block.backward(dx)
        self.dW_emb.fill(0)
        np.add.at(self.dW_emb, self.input_ids, dx)

    def get_params_and_grads(self):
        pairs = [(self.W_emb, self.dW_emb), (self.W_head, self.dW_head), (self.b_head, self.db_head)]
        for block in self.blocks:
            if isinstance(block, MultiHeadAttentionNumPy):
                pairs.extend([(block.WQ, block.dWQ), (block.WK, block.dWK), (block.WV, block.dWV), (block.WO, block.dWO), (block.bO, block.dbO)])
            elif isinstance(block, TransformerBlockNumPy):
                attn = block.attn
                pairs.extend([(attn.WQ, attn.dWQ), (attn.WK, attn.dWK), (attn.WV, attn.dWV), (attn.WO, attn.dWO), (attn.bO, attn.dbO)])
                if block.use_ln and block.ln1:
                    pairs.extend([(block.ln1.gamma, block.ln1.dgamma), (block.ln1.beta, block.ln1.dbeta)])
                if block.use_ffn:
                    ffn = block.ffn
                    pairs.extend([(ffn.W1, ffn.dW1), (ffn.b1, ffn.db1), (ffn.W2, ffn.dW2), (ffn.b2, ffn.db2)])
                    if block.use_ln and block.ln2:
                        pairs.extend([(block.ln2.gamma, block.ln2.dgamma), (block.ln2.beta, block.ln2.dbeta)])
        return pairs




def assert_tensor_shapes(model, input_ids, mask):
    B, T = input_ids.shape
    d_model, vocab_size = model.d_model, model.vocab_size
    emb = model.W_emb[input_ids] + model.pos_enc[:T]
    assert emb.shape == (B, T, d_model), f"Embedding shape mismatch: {emb.shape}"
    logits, x_final = model.forward(input_ids)
    assert logits.shape == (B, T, vocab_size), f"Logits shape mismatch: {logits.shape}"
    loss, dlogits, probs = compute_cross_entropy_loss(logits, input_ids, mask)
    assert dlogits.shape == (B, T, vocab_size), f"dLogits shape mismatch: {dlogits.shape}"
    print(f"[OK] All shape assertions passed for Model Variant {model.model_id}.")

def check_attention_logic(model, input_ids):
    logits, x_final = model.forward(input_ids)
    # Check if attention sum to 1
    for block in model.blocks:
        attn = block.attn if hasattr(block, 'attn') else (block if isinstance(block, MultiHeadAttentionNumPy) else None)
        if attn and attn.cache:
            weights = attn.cache[4]
            sums = np.sum(weights, axis=-1)
            assert np.allclose(sums, 1.0), "Attention weights do not sum to 1."
            # Check causal masking (future positions are zero)
            B, h, T, T2 = weights.shape
            upper_tri = np.triu(weights[0, 0], k=1)
            assert np.allclose(upper_tri, 0.0), "Future positions have non-zero attention."
    print("[OK] Attention logic and causal masking verified.")

def finite_difference_gradient_check(model, input_ids, targets, mask, eps=1e-5, max_check_params=10, threshold=1e-3):
    print(f"\n--- Finite-Difference Gradient Check (Model {model.model_id}) ---")
    logits, x_final = model.forward(input_ids)
    loss, dlogits, probs = compute_cross_entropy_loss(logits, targets, mask)
    model.backward(dlogits, x_final)

    param_grad_pairs = model.get_params_and_grads()
    all_passed = True
    
    for idx, (param, grad_analytical) in enumerate(param_grad_pairs):
        param_flat = param.ravel()
        grad_flat = grad_analytical.ravel()
        sample_indices = np.random.choice(len(param_flat), size=min(max_check_params, len(param_flat)), replace=False)
        max_rel_error = 0.0
        max_abs_error = 0.0

        for i in sample_indices:
            orig_val = param_flat[i]
            param_flat[i] = orig_val + eps
            loss_plus = compute_cross_entropy_loss(model.forward(input_ids)[0], targets, mask)[0]
            param_flat[i] = orig_val - eps
            loss_minus = compute_cross_entropy_loss(model.forward(input_ids)[0], targets, mask)[0]
            param_flat[i] = orig_val

            grad_num = (loss_plus - loss_minus) / (2.0 * eps)
            grad_ana = grad_flat[i]
            
            abs_err = np.abs(grad_ana - grad_num)
            rel_err = abs_err / (np.abs(grad_ana) + np.abs(grad_num) + 1e-8)
            
            max_abs_error = max(max_abs_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)

        # Allow slight leniency if absolute error is incredibly small (e.g., near zero gradients)
        status = "PASSED" if (max_rel_error < threshold or max_abs_error < 1e-7) else "FAILED"
        if status == "FAILED":
            all_passed = False
        print(f"Param #{idx+1:2d} {str(param.shape):<10s} | Max Rel: {max_rel_error:.2e} | Max Abs: {max_abs_error:.2e} -> {status}")

    return all_passed

np.random.seed(42)
dummy_x = np.random.randint(1, VOCAB_SIZE, size=(2, 6))
dummy_y = np.random.randint(1, VOCAB_SIZE, size=(2, 6))
dummy_mask = np.ones((2, 6), dtype=np.float64)

test_model = ModularTinyTransformer("D", vocab_size=VOCAB_SIZE, d_model=24, d_ff=96, max_len=24, seed=42)
assert_tensor_shapes(test_model, dummy_x, dummy_mask)
check_attention_logic(test_model, dummy_x)
finite_difference_gradient_check(test_model, dummy_x, dummy_y, dummy_mask, eps=1e-5, threshold=1e-3)




def clip_gradients(param_grad_pairs, max_norm=1.0):
    total_norm_sq = sum(np.sum(grad ** 2) for _, grad in param_grad_pairs)
    total_norm = np.sqrt(total_norm_sq)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        for _, grad in param_grad_pairs:
            grad *= scale

def evaluate_metrics(model, batches):
    losses, correct, total = [], 0, 0
    all_preds = []
    all_targets = []
    for batch in batches:
        X, Y, mask = batch["X"], batch["Y"], batch["mask"]
        logits, _ = model.forward(X)
        loss, _, probs = compute_cross_entropy_loss(logits, Y, mask)
        losses.append(loss)
        preds = np.argmax(probs, axis=-1)
        valid_mask = (mask > 0)
        
        correct += np.sum((preds == Y) & valid_mask)
        total += np.sum(valid_mask)
        
        # Collect for macro F1
        all_preds.extend(preds[valid_mask].tolist())
        all_targets.extend(Y[valid_mask].tolist())
        
    acc = (correct / total) if total > 0 else 0.0
    
    # Calculate Macro F1
    from collections import defaultdict
    classes = set(all_targets)
    f1_sum = 0
    for cls in classes:
        tp = sum(1 for p, t in zip(all_preds, all_targets) if p == cls and t == cls)
        fp = sum(1 for p, t in zip(all_preds, all_targets) if p == cls and t != cls)
        fn = sum(1 for p, t in zip(all_preds, all_targets) if p != cls and t == cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_sum += f1
        
    macro_f1 = (f1_sum / len(classes)) if len(classes) > 0 else 0.0
    
    return np.mean(losses), acc * 100.0, macro_f1

def train_single_run(model_id, train_batches, val_batches, test_batches, seed, lr=0.03, max_epochs=150, patience=20, d_ff=96):
    model = ModularTinyTransformer(model_id, vocab_size=VOCAB_SIZE, d_model=24, d_ff=d_ff, max_len=24, seed=seed)
    best_val_loss, best_weights, patience_counter = float('inf'), None, 0
    
    start_time = time.time()
    for epoch in range(max_epochs):
        for batch in train_batches:
            X, Y, mask = batch["X"], batch["Y"], batch["mask"]
            logits, x_final = model.forward(X)
            loss, dlogits, _ = compute_cross_entropy_loss(logits, Y, mask)
            model.backward(dlogits, x_final)
            param_grad_pairs = model.get_params_and_grads()
            clip_gradients(param_grad_pairs, max_norm=1.0)
            for param, grad in param_grad_pairs:
                param -= lr * grad
                
        val_loss, val_acc, val_f1 = evaluate_metrics(model, val_batches)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = [param.copy() for param, _ in model.get_params_and_grads()]
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best weights
    param_grad_pairs = model.get_params_and_grads()
    for idx, (param, _) in enumerate(param_grad_pairs):
        param[:] = best_weights[idx]
        
    train_loss, train_acc, train_f1 = evaluate_metrics(model, train_batches)
    test_loss, test_acc, test_f1 = evaluate_metrics(model, test_batches)
    
    return {
        "model_id": model_id, "seed": seed, "d_ff": d_ff,
        "epochs": epoch + 1, "time": time.time() - start_time,
        "train_loss": train_loss, "val_loss": best_val_loss, "test_loss": test_loss,
        "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
        "test_f1": test_f1, "params": model.get_num_params()
    }




seeds = [7, 19, 42, 73, 101]
architectures = ["A", "B", "C", "D", "D-1", "D-no-FFN", "D-no-LN", "D-no-res"]

train_batches = create_dataset_batches(splits["train"])
val_batches = create_dataset_batches(splits["val"])
test_batches = create_dataset_batches(splits["test"])

results = []
print("Starting Architecture Benchmark...")
for arch in architectures:
    print(f"\nTraining Model {arch}")
    for seed in seeds:
        res = train_single_run(arch, train_batches, val_batches, test_batches, seed=seed)
        results.append(res)
        print(f"  Seed {seed}: Test Acc {res['test_acc']:.2f}%, Test Loss {res['test_loss']:.4f}")

# Summarize
print("\n--- Architecture Summary ---")
for arch in architectures:
    arch_res = [r for r in results if r['model_id'] == arch]
    mean_acc = np.mean([r['test_acc'] for r in arch_res])
    std_acc = np.std([r['test_acc'] for r in arch_res])
    mean_loss = np.mean([r['test_loss'] for r in arch_res])
    print(f"Model {arch:10s} | Params: {arch_res[0]['params']} | Loss: {mean_loss:.4f} | Acc: {mean_acc:.2f}% ± {std_acc:.2f}%")




ffn_widths = [24, 48, 96, 192]
ffn_results = []
print("\nStarting FFN-Width Experiment (Model D-1)...")
for width in ffn_widths:
    print(f"\nWidth {width}")
    for seed in seeds:
        res = train_single_run("D-1", train_batches, val_batches, test_batches, seed=seed, d_ff=width)
        ffn_results.append(res)

import matplotlib.pyplot as plt
import os
os.makedirs("projects/week 5/visualizations", exist_ok=True)

width_means = []
gaps = []
for w in ffn_widths:
    w_res = [r for r in ffn_results if r['d_ff'] == w]
    mean_test = np.mean([r['test_loss'] for r in w_res])
    mean_train = np.mean([r['train_loss'] for r in w_res])
    width_means.append(mean_test)
    gaps.append(mean_test - mean_train)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(ffn_widths, width_means, marker='o')
plt.title("Test Loss vs d_ff")
plt.xlabel("FFN Width")
plt.ylabel("Test Loss")

plt.subplot(1, 2, 2)
plt.plot(ffn_widths, gaps, marker='o', color='red')
plt.title("Generalization Gap (Test - Train Loss)")
plt.xlabel("FFN Width")
plt.ylabel("Loss Gap")
plt.tight_layout()
plt.savefig("projects/week 5/visualizations/ffn_width_experiment.png")
plt.show()

