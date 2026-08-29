"""
numpy_transformer_suite.py
===========================
Pure-NumPy implementation of all Transformer building blocks.

Design notes:
- All arithmetic uses float64 for numerical stability (important for gradient checks).
- Backward passes are fully analytically derived — no autograd.
- Supports model variants A, B, C, D, D-1, D-no-FFN, D-no-LN, D-no-res.
- Training history (loss, accuracy, grad_norm) is recorded per epoch.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def sinusoidal_positional_encoding(seq_len, d_model):
    """Sinusoidal positional encoding. Returns (seq_len, d_model) float64 array."""
    pe = np.zeros((seq_len, d_model), dtype=np.float64)
    position = np.arange(0, seq_len, dtype=np.float64)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, d_model, 2, dtype=np.float64) * -(np.log(10000.0) / d_model)
    )
    pe[:, 0::2] = np.sin(position * div_term)
    if d_model % 2 == 1:
        pe[:, 1::2] = np.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = np.cos(position * div_term)
    return pe


def compute_cross_entropy_loss(logits, targets, mask):
    """
    Masked cross-entropy loss.

    Parameters
    ----------
    logits  : (B, T, V) float64
    targets : (B, T)    int64
    mask    : (B, T)    float32 — 1.0 for valid positions, 0.0 for padding

    Returns
    -------
    loss    : scalar
    dlogits : (B, T, V) gradient w.r.t. logits
    probs   : (B, T, V) softmax probabilities
    """
    probs = softmax(logits, axis=-1)
    B, T, V = logits.shape

    batch_idx = np.arange(B)[:, np.newaxis]
    time_idx = np.arange(T)[np.newaxis, :]

    target_probs = probs[batch_idx, time_idx, targets]
    log_probs = np.log(np.maximum(target_probs, 1e-12))

    masked_loss = -log_probs * mask
    total_valid = np.sum(mask)
    loss = np.sum(masked_loss) / np.maximum(total_valid, 1.0)

    # Gradient: softmax - one_hot, masked and normalized
    dlogits = probs.copy()
    dlogits[batch_idx, time_idx, targets] -= 1.0
    dlogits = dlogits * mask[:, :, np.newaxis] / np.maximum(total_valid, 1.0)

    return loss, dlogits, probs


# ---------------------------------------------------------------------------
# Layer Normalization
# ---------------------------------------------------------------------------

class LayerNormNumPy:
    """
    Pre-LN Layer Normalization.
    Formula: y = gamma * (x - mu) / sigma + beta
    
    Acts as a stabilizer. When multiplying many values together, variance can 
    explode to infinity or vanish to zero. This component mathematically 
    normalizes the values back into a safe range to maintain stability.
    """
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

    def get_params_and_grads(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForwardNumPy:
    """
    Position-wise Feed-Forward Network.
    FFN(x) = ReLU(x W1 + b1) W2 + b2
    Shapes: (B,T,d_model) -> (B,T,d_ff) -> (B,T,d_model)
    
    Acts as the deep processor. Once the Attention mechanism finds a pattern, 
    this component processes it using a ReLU activation function to model 
    complex, non-linear rules.
    """
    def __init__(self, d_model, d_ff, rng):
        self.d_model = d_model
        self.d_ff = d_ff
        # Xavier uniform initialization
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
        h1 = np.matmul(x, self.W1) + self.b1     # (B,T,d_ff)
        a1 = np.maximum(0.0, h1)                  # ReLU
        out = np.matmul(a1, self.W2) + self.b2    # (B,T,d_model)
        self.cache = (x, h1, a1)
        return out

    def backward(self, dout):
        x, h1, a1 = self.cache
        self.dW2 = np.matmul(a1.reshape(-1, self.d_ff).T,
                              dout.reshape(-1, self.d_model))
        self.db2 = np.sum(dout, axis=(0, 1))

        da1 = np.matmul(dout, self.W2.T)
        dh1 = da1 * (h1 > 0.0)                   # ReLU gradient

        self.dW1 = np.matmul(x.reshape(-1, self.d_model).T,
                              dh1.reshape(-1, self.d_ff))
        self.db1 = np.sum(dh1, axis=(0, 1))

        dx = np.matmul(dh1, self.W1.T)
        return dx

    def get_params_and_grads(self):
        return [(self.W1, self.dW1), (self.b1, self.db1),
                (self.W2, self.dW2), (self.b2, self.db2)]


# ---------------------------------------------------------------------------
# Multi-Head Causal Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadAttentionNumPy:
    """
    Causal multi-head self-attention.
    Each head has dimension d_k = d_model // num_heads.
    Future positions are masked with -1e9 before softmax.
    
    Acts as the context engine. It evaluates the history of the sequence and 
    determines which past events are most relevant. It uses a causal mask 
    to prevent the model from attending to future tokens.
    """
    def __init__(self, d_model, num_heads, rng):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        limit = np.sqrt(6.0 / (d_model + d_model))
        self.WQ = rng.uniform(-limit, limit, (d_model, d_model)).astype(np.float64)
        self.WK = rng.uniform(-limit, limit, (d_model, d_model)).astype(np.float64)
        self.WV = rng.uniform(-limit, limit, (d_model, d_model)).astype(np.float64)
        self.WO = rng.uniform(-limit, limit, (d_model, d_model)).astype(np.float64)
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

        attn_weights = softmax(scores, axis=-1)   # (B, h, T, T)
        context = np.matmul(attn_weights, V)      # (B, h, T, d_k)

        context_concat = context.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        out = np.matmul(context_concat, self.WO) + self.bO

        self.cache = (x, Q, K, V, attn_weights, context_concat)
        return out

    def backward(self, dout):
        x, Q, K, V, attn_weights, context_concat = self.cache
        B, T, _ = x.shape
        h, d_k = self.num_heads, self.d_k

        self.dWO = np.matmul(context_concat.reshape(-1, self.d_model).T,
                              dout.reshape(-1, self.d_model))
        self.dbO = np.sum(dout, axis=(0, 1))

        dcontext_concat = np.matmul(dout, self.WO.T)
        dcontext = dcontext_concat.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)

        dV = np.matmul(attn_weights.transpose(0, 1, 3, 2), dcontext)
        dattn_weights = np.matmul(dcontext, V.transpose(0, 1, 3, 2))

        # Softmax backward
        dscores = attn_weights * (
            dattn_weights - np.sum(dattn_weights * attn_weights, axis=-1, keepdims=True)
        )
        dscores = dscores / np.sqrt(d_k)

        dQ = np.matmul(dscores, K)
        dK = np.matmul(dscores.transpose(0, 1, 3, 2), Q)

        dQ_flat = dQ.transpose(0, 2, 1, 3).reshape(B * T, self.d_model)
        dK_flat = dK.transpose(0, 2, 1, 3).reshape(B * T, self.d_model)
        dV_flat = dV.transpose(0, 2, 1, 3).reshape(B * T, self.d_model)
        x_flat = x.reshape(B * T, self.d_model)

        self.dWQ = np.matmul(x_flat.T, dQ_flat)
        self.dWK = np.matmul(x_flat.T, dK_flat)
        self.dWV = np.matmul(x_flat.T, dV_flat)

        dx = (np.matmul(dQ_flat, self.WQ.T)
              + np.matmul(dK_flat, self.WK.T)
              + np.matmul(dV_flat, self.WV.T))
        return dx.reshape(B, T, self.d_model)

    def get_params_and_grads(self):
        return [(self.WQ, self.dWQ), (self.WK, self.dWK),
                (self.WV, self.dWV), (self.WO, self.dWO),
                (self.bO, self.dbO)]

    def get_last_attention_weights(self):
        """Returns (B, H, T, T) attention weights from last forward pass."""
        return self.cache[4] if self.cache else None


# ---------------------------------------------------------------------------
# Transformer Block (Pre-LN)
# ---------------------------------------------------------------------------

class TransformerBlockNumPy:
    """
    Pre-LN Transformer block.

    Forward:
        norm1  = LayerNorm(x)
        attn   = MultiHeadAttention(norm1)
        x1     = x + attn              [residual 1]
        norm2  = LayerNorm(x1)
        ffn    = FFN(norm2)
        x2     = x1 + ffn              [residual 2]

    Combines LayerNorm, MultiHeadAttention, and FeedForward components. 
    It utilizes residual connections (x + attn) to allow gradients and 
    unprocessed features to bypass heavy processing, preserving original 
    sequence information in deeper layers.
    
    Ablation flags:
        use_ffn  : if False, omit FFN sub-layer
        use_ln   : if False, omit both LayerNorm layers
        use_res  : if False, omit both residual additions
    """
    def __init__(self, d_model, num_heads, d_ff, rng,
                 use_ffn=True, use_ln=True, use_res=True):
        self.use_ffn = use_ffn
        self.use_ln = use_ln
        self.use_res = use_res

        self.attn = MultiHeadAttentionNumPy(d_model, num_heads, rng)
        self.ln1 = LayerNormNumPy(d_model) if use_ln else None

        if use_ffn:
            self.ffn = FeedForwardNumPy(d_model, d_ff, rng)
            self.ln2 = LayerNormNumPy(d_model) if use_ln else None
        else:
            self.ffn = None
            self.ln2 = None

        self._cache_x = None  # for residual backward

    def forward(self, x):
        # --- Sub-layer 1: Attention ---
        norm1 = self.ln1.forward(x) if self.use_ln else x
        attn_out = self.attn.forward(norm1)
        x1 = (x + attn_out) if self.use_res else attn_out
        self._cache_x = x

        # --- Sub-layer 2: FFN ---
        if self.use_ffn:
            norm2 = self.ln2.forward(x1) if self.use_ln else x1
            ffn_out = self.ffn.forward(norm2)
            x2 = (x1 + ffn_out) if self.use_res else ffn_out
        else:
            x2 = x1

        return x2

    def backward(self, dout):
        x = self._cache_x

        # --- Sub-layer 2 backward ---
        if self.use_ffn:
            # dout flows through residual: dx1 += dout (from residual addition)
            dffn_out = dout
            dx_norm2 = self.ffn.backward(dffn_out)
            dnorm2 = self.ln2.backward(dx_norm2) if self.use_ln else dx_norm2
            dx1 = (dout + dnorm2) if self.use_res else dnorm2
        else:
            dx1 = dout

        # --- Sub-layer 1 backward ---
        dattn_out = dx1
        dx_norm1 = self.attn.backward(dattn_out)
        dnorm1 = self.ln1.backward(dx_norm1) if self.use_ln else dx_norm1
        dx = (dx1 + dnorm1) if self.use_res else dnorm1

        return dx

    def get_params_and_grads(self):
        pairs = self.attn.get_params_and_grads()
        if self.use_ln and self.ln1:
            pairs += self.ln1.get_params_and_grads()
        if self.use_ffn and self.ffn:
            pairs += self.ffn.get_params_and_grads()
            if self.use_ln and self.ln2:
                pairs += self.ln2.get_params_and_grads()
        return pairs


# ---------------------------------------------------------------------------
# Unified Modular Transformer Model
# ---------------------------------------------------------------------------

class ModularTinyTransformer:
    """
    Unified Pure-NumPy next-token-prediction model.

    Supported model_ids:
        A       : Embedding + PE + Linear head (no attention, no FFN)
        B       : A + 1-head causal self-attention
        C       : A + 4-head causal self-attention
        D       : A + 2 Pre-LN Transformer blocks (4 heads, FFN, LN, residual)
        D-1     : A + 1 Pre-LN Transformer block
        D-no-FFN: A + 2 Transformer blocks WITHOUT FFN
        D-no-LN : A + 2 Transformer blocks WITHOUT LayerNorm
        D-no-res: A + 2 Transformer blocks WITHOUT residual connections
    """

    MODEL_DESCRIPTIONS = {
        "A":        "Embedding + Positional Encoding + Linear head",
        "B":        "Model A + 1-head causal self-attention",
        "C":        "Model A + 4-head causal self-attention",
        "D":        "Model A + 2 Pre-LN Transformer blocks",
        "D-1":      "Model A + 1 Pre-LN Transformer block",
        "D-no-FFN": "2 Transformer blocks WITHOUT Feed-Forward Networks",
        "D-no-LN":  "2 Transformer blocks WITHOUT LayerNorm",
        "D-no-res": "2 Transformer blocks WITHOUT residual connections",
    }

    def __init__(self, model_id, vocab_size, d_model=24, d_ff=96, max_len=20, seed=42):
        if model_id not in self.MODEL_DESCRIPTIONS:
            raise ValueError(f"Unknown model_id: {model_id!r}. "
                             f"Must be one of {list(self.MODEL_DESCRIPTIONS)}")

        self.model_id = model_id
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.max_len = max_len

        rng = np.random.RandomState(seed)

        # Embedding & sinusoidal PE (float64)
        limit_emb = np.sqrt(6.0 / (vocab_size + d_model))
        self.W_emb = rng.uniform(-limit_emb, limit_emb,
                                 (vocab_size, d_model)).astype(np.float64)
        self.pos_enc = sinusoidal_positional_encoding(max_len, d_model)
        self.dW_emb = np.zeros_like(self.W_emb)

        # Architecture
        self.blocks = []
        if model_id == "A":
            pass
        elif model_id == "B":
            self.blocks.append(MultiHeadAttentionNumPy(d_model, num_heads=1, rng=rng))
        elif model_id == "C":
            self.blocks.append(MultiHeadAttentionNumPy(d_model, num_heads=4, rng=rng))
        elif model_id == "D":
            self.blocks.append(TransformerBlockNumPy(d_model, 4, d_ff, rng))
            self.blocks.append(TransformerBlockNumPy(d_model, 4, d_ff, rng))
        elif model_id == "D-1":
            self.blocks.append(TransformerBlockNumPy(d_model, 4, d_ff, rng))
        elif model_id == "D-no-FFN":
            self.blocks.append(TransformerBlockNumPy(d_model, 4, d_ff, rng, use_ffn=False))
            self.blocks.append(TransformerBlockNumPy(d_model, 4, d_ff, rng, use_ffn=False))
        elif model_id == "D-no-LN":
            self.blocks.append(TransformerBlockNumPy(d_model, 4, d_ff, rng, use_ln=False))
            self.blocks.append(TransformerBlockNumPy(d_model, 4, d_ff, rng, use_ln=False))
        elif model_id == "D-no-res":
            self.blocks.append(TransformerBlockNumPy(d_model, 4, d_ff, rng, use_res=False))
            self.blocks.append(TransformerBlockNumPy(d_model, 4, d_ff, rng, use_res=False))

        # Output head
        limit_head = np.sqrt(6.0 / (d_model + vocab_size))
        self.W_head = rng.uniform(-limit_head, limit_head,
                                  (d_model, vocab_size)).astype(np.float64)
        self.b_head = np.zeros(vocab_size, dtype=np.float64)
        self.dW_head = np.zeros_like(self.W_head)
        self.db_head = np.zeros_like(self.b_head)

        self._input_ids = None  # saved for embedding backward

    def forward(self, input_ids):
        """
        Parameters
        ----------
        input_ids : (B, T) int64 array

        Returns
        -------
        logits  : (B, T, V)
        x_final : (B, T, d_model)  — needed for backward
        """
        self._input_ids = input_ids
        B, T = input_ids.shape

        x = self.W_emb[input_ids].astype(np.float64)   # (B, T, d_model)
        x = x + self.pos_enc[:T]

        for block in self.blocks:
            x = block.forward(x)

        logits = np.matmul(x, self.W_head) + self.b_head   # (B, T, V)
        return logits, x

    def backward(self, dlogits, x_final):
        """
        Parameters
        ----------
        dlogits : (B, T, V) — gradient from loss
        x_final : (B, T, d_model) — final hidden states
        """
        self.dW_head = np.matmul(
            x_final.reshape(-1, self.d_model).T,
            dlogits.reshape(-1, self.vocab_size)
        )
        self.db_head = np.sum(dlogits, axis=(0, 1))

        dx = np.matmul(dlogits, self.W_head.T)

        for block in reversed(self.blocks):
            dx = block.backward(dx)

        # Scatter gradients back to embedding rows
        self.dW_emb.fill(0.0)
        np.add.at(self.dW_emb, self._input_ids, dx)

    def get_params_and_grads(self):
        """Returns list of (param_array, grad_array) tuples for the optimizer."""
        pairs = [
            (self.W_emb, self.dW_emb),
            (self.W_head, self.dW_head),
            (self.b_head, self.db_head),
        ]
        for block in self.blocks:
            if isinstance(block, MultiHeadAttentionNumPy):
                pairs += block.get_params_and_grads()
            elif isinstance(block, TransformerBlockNumPy):
                pairs += block.get_params_and_grads()
        return pairs

    def count_parameters(self):
        return sum(p.size for p, _ in self.get_params_and_grads())

    def get_attention_weights(self, layer_idx=0):
        """
        Returns attention weights from a given block (0-indexed).
        Returns None if block does not have attention or no forward pass done.
        """
        if layer_idx >= len(self.blocks):
            return None
        block = self.blocks[layer_idx]
        if isinstance(block, TransformerBlockNumPy):
            return block.attn.get_last_attention_weights()
        elif isinstance(block, MultiHeadAttentionNumPy):
            return block.get_last_attention_weights()
        return None
