import numpy as np

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def sinusoidal_positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    position = np.arange(0, seq_len, dtype=np.float32)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

class LayerNormNumPy:
    def __init__(self, d_model, eps=1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)
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
        # dout shape: (B, T, d_model)
        self.dgamma = np.sum(dout * x_norm, axis=(0, 1))
        self.dbeta = np.sum(dout, axis=(0, 1))
        
        dx_norm = dout * self.gamma
        N = self.d_model
        # Backprop through layer normalization
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
        # Xavier uniform initialization
        limit1 = np.sqrt(6.0 / (d_model + d_ff))
        self.W1 = rng.uniform(-limit1, limit1, (d_model, d_ff)).astype(np.float32)
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        
        limit2 = np.sqrt(6.0 / (d_ff + d_model))
        self.W2 = rng.uniform(-limit2, limit2, (d_ff, d_model)).astype(np.float32)
        self.b2 = np.zeros(d_model, dtype=np.float32)

        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)
        self.cache = None

    def forward(self, x):
        h1 = np.matmul(x, self.W1) + self.b1
        a1 = np.maximum(0, h1) # ReLU activation
        out = np.matmul(a1, self.W2) + self.b2
        self.cache = (x, h1, a1)
        return out

    def backward(self, dout):
        x, h1, a1 = self.cache
        # dout shape: (B, T, d_model)
        self.dW2 = np.matmul(a1.reshape(-1, self.d_ff).T, dout.reshape(-1, self.d_model))
        self.db2 = np.sum(dout, axis=(0, 1))

        da1 = np.matmul(dout, self.W2.T)
        dh1 = da1 * (h1 > 0)

        self.dW1 = np.matmul(x.reshape(-1, self.d_model).T, dh1.reshape(-1, self.d_ff))
        self.db1 = np.sum(dh1, axis=(0, 1))

        dx = np.matmul(dh1, self.W1.T)
        return dx


class MultiHeadAttentionNumPy:
    def __init__(self, d_model, num_heads, rng):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        limit = np.sqrt(6.0 / (d_model + d_model))
        self.WQ = rng.uniform(-limit, limit, (d_model, d_model)).astype(np.float32)
        self.WK = rng.uniform(-limit, limit, (d_model, d_model)).astype(np.float32)
        self.WV = rng.uniform(-limit, limit, (d_model, d_model)).astype(np.float32)
        self.WO = rng.uniform(-limit, limit, (d_model, d_model)).astype(np.float32)
        self.bO = np.zeros(d_model, dtype=np.float32)

        self.dWQ = np.zeros_like(self.WQ)
        self.dWK = np.zeros_like(self.WK)
        self.dWV = np.zeros_like(self.WV)
        self.dWO = np.zeros_like(self.WO)
        self.dbO = np.zeros_like(self.bO)
        self.cache = None

    def forward(self, x):
        B, T, _ = x.shape
        h = self.num_heads
        d_k = self.d_k

        Q = np.matmul(x, self.WQ).reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
        K = np.matmul(x, self.WK).reshape(B, T, h, d_k).transpose(0, 2, 1, 3)
        V = np.matmul(x, self.WV).reshape(B, T, h, d_k).transpose(0, 2, 1, 3)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        
        # Causal mask
        causal_mask = np.triu(np.ones((T, T), dtype=np.float32), k=1) * -1e9
        scores = scores + causal_mask[np.newaxis, np.newaxis, :, :]

        attn_weights = softmax(scores, axis=-1)
        context = np.matmul(attn_weights, V) # (B, h, T, d_k)

        context_concat = context.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        out = np.matmul(context_concat, self.WO) + self.bO

        self.cache = (x, Q, K, V, attn_weights, context_concat)
        return out

    def backward(self, dout):
        x, Q, K, V, attn_weights, context_concat = self.cache
        B, T, _ = x.shape
        h = self.num_heads
        d_k = self.d_k

        self.dWO = np.matmul(context_concat.reshape(-1, self.d_model).T, dout.reshape(-1, self.d_model))
        self.dbO = np.sum(dout, axis=(0, 1))

        dcontext_concat = np.matmul(dout, self.WO.T)
        dcontext = dcontext_concat.reshape(B, T, h, d_k).transpose(0, 2, 1, 3)

        # dcontext = attn_weights @ dV + d_attn_weights @ V
        dV = np.matmul(attn_weights.transpose(0, 1, 3, 2), dcontext)
        dattn_weights = np.matmul(dcontext, V.transpose(0, 1, 3, 2))

        # Softmax backward
        dscores = attn_weights * (dattn_weights - np.sum(dattn_weights * attn_weights, axis=-1, keepdims=True))
        dscores = dscores / np.sqrt(d_k)

        dQ = np.matmul(dscores, K)
        dK = np.matmul(dscores.transpose(0, 1, 3, 2), Q)

        # Reshape dQ, dK, dV back to (B, T, d_model)
        dQ_flat = dQ.transpose(0, 2, 1, 3).reshape(B * T, self.d_model)
        dK_flat = dK.transpose(0, 2, 1, 3).reshape(B * T, self.d_model)
        dV_flat = dV.transpose(0, 2, 1, 3).reshape(B * T, self.d_model)
        x_flat = x.reshape(B * T, self.d_model)

        self.dWQ = np.matmul(x_flat.T, dQ_flat)
        self.dWK = np.matmul(x_flat.T, dK_flat)
        self.dWV = np.matmul(x_flat.T, dV_flat)

        dx = np.matmul(dQ_flat, self.WQ.T) + np.matmul(dK_flat, self.WK.T) + np.matmul(dV_flat, self.WV.T)
        return dx.reshape(B, T, self.d_model)


class TransformerBlockNumPy:
    def __init__(self, d_model, num_heads, d_ff, rng, use_ffn=True, use_ln=True):
        self.use_ffn = use_ffn
        self.use_ln = use_ln
        
        self.attn = MultiHeadAttentionNumPy(d_model, num_heads, rng)
        self.ln1 = LayerNormNumPy(d_model) if use_ln else None
        
        if use_ffn:
            self.ffn = FeedForwardNumPy(d_model, d_ff, rng)
            self.ln2 = LayerNormNumPy(d_model) if use_ln else None

    def forward(self, x):
        # Pre-LN Block Structure
        norm1 = self.ln1.forward(x) if self.use_ln else x
        attn_out = self.attn.forward(norm1)
        x1 = x + attn_out

        if self.use_ffn:
            norm2 = self.ln2.forward(x1) if self.use_ln else x1
            ffn_out = self.ffn.forward(norm2)
            x2 = x1 + ffn_out
        else:
            x2 = x1

        return x2

    def backward(self, dout):
        if self.use_ffn:
            dffn_out = dout
            dx1_from_ffn = self.ffn.backward(dffn_out)
            dnorm2 = self.ln2.backward(dx1_from_ffn) if self.use_ln else dx1_from_ffn
            dx1 = dout + dnorm2
        else:
            dx1 = dout

        dattn_out = dx1
        dnorm1 = self.attn.backward(dattn_out)
        dx_from_attn = self.ln1.backward(dnorm1) if self.use_ln else dnorm1
        dx = dx1 + dx_from_attn
        return dx


class ModularTinyTransformer:
    """
    Unified Pure-NumPy Model supporting all 7 Model Matrix variants:
    A: embedding + position + linear head
    B: A + 1-head causal attention
    C: A + 4-head causal attention
    D: A + 2 pre-LN Transformer blocks
    D-1: 1 Transformer block
    D-no-FFN: 2 blocks without FFN
    D-no-LN: 2 blocks without LayerNorm
    """
    def __init__(self, model_id, vocab_size, d_model=24, d_ff=96, max_len=10, seed=42):
        self.model_id = model_id
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.max_len = max_len
        
        rng = np.random.RandomState(seed)

        # Embedding & Positional Encoding
        limit_emb = np.sqrt(6.0 / (vocab_size + d_model))
        self.W_emb = rng.uniform(-limit_emb, limit_emb, (vocab_size, d_model)).astype(np.float32)
        self.pos_enc = sinusoidal_positional_encoding(max_len, d_model)
        self.dW_emb = np.zeros_like(self.W_emb)

        # Parse Architecture based on model_id
        self.blocks = []
        if model_id == "A":
            pass
        elif model_id == "B":
            self.blocks.append(MultiHeadAttentionNumPy(d_model, num_heads=1, rng=rng))
        elif model_id == "C":
            self.blocks.append(MultiHeadAttentionNumPy(d_model, num_heads=4, rng=rng))
        elif model_id == "D":
            self.blocks.append(TransformerBlockNumPy(d_model, num_heads=4, d_ff=d_ff, rng=rng, use_ffn=True, use_ln=True))
            self.blocks.append(TransformerBlockNumPy(d_model, num_heads=4, d_ff=d_ff, rng=rng, use_ffn=True, use_ln=True))
        elif model_id == "D-1":
            self.blocks.append(TransformerBlockNumPy(d_model, num_heads=4, d_ff=d_ff, rng=rng, use_ffn=True, use_ln=True))
        elif model_id == "D-no-FFN":
            self.blocks.append(TransformerBlockNumPy(d_model, num_heads=4, d_ff=d_ff, rng=rng, use_ffn=False, use_ln=True))
            self.blocks.append(TransformerBlockNumPy(d_model, num_heads=4, d_ff=d_ff, rng=rng, use_ffn=False, use_ln=True))
        elif model_id == "D-no-LN":
            self.blocks.append(TransformerBlockNumPy(d_model, num_heads=4, d_ff=d_ff, rng=rng, use_ffn=True, use_ln=False))
            self.blocks.append(TransformerBlockNumPy(d_model, num_heads=4, d_ff=d_ff, rng=rng, use_ffn=True, use_ln=False))
        else:
            raise ValueError(f"Unknown model_id: {model_id}")

        # Linear Output Head
        limit_head = np.sqrt(6.0 / (d_model + vocab_size))
        self.W_head = rng.uniform(-limit_head, limit_head, (d_model, vocab_size)).astype(np.float32)
        self.b_head = np.zeros(vocab_size, dtype=np.float32)
        self.dW_head = np.zeros_like(self.W_head)
        self.db_head = np.zeros_like(self.b_head)
        
        self.input_ids = None

    def forward(self, input_ids):
        self.input_ids = input_ids
        B, T = input_ids.shape
        
        emb = self.W_emb[input_ids] # (B, T, d_model)
        x = emb + self.pos_enc[:T]

        for block in self.blocks:
            x = block.forward(x)

        logits = np.matmul(x, self.W_head) + self.b_head
        return logits, x

    def backward(self, dlogits, x_final):
        B, T, _ = dlogits.shape
        
        self.dW_head = np.matmul(x_final.reshape(-1, self.d_model).T, dlogits.reshape(-1, self.vocab_size))
        self.db_head = np.sum(dlogits, axis=(0, 1))

        dx = np.matmul(dlogits, self.W_head.T)

        for block in reversed(self.blocks):
            dx = block.backward(dx)

        # Backprop to Embedding Matrix
        self.dW_emb.fill(0)
        np.add.at(self.dW_emb, self.input_ids, dx)

    def get_params_and_grads(self):
        """Returns list of (param, grad) tuples for SGD optimizer."""
        pairs = [
            (self.W_emb, self.dW_emb),
            (self.W_head, self.dW_head),
            (self.b_head, self.db_head)
        ]
        
        for block in self.blocks:
            if isinstance(block, MultiHeadAttentionNumPy):
                pairs.extend([
                    (block.WQ, block.dWQ), (block.WK, block.dWK),
                    (block.WV, block.dWV), (block.WO, block.dWO),
                    (block.bO, block.dbO)
                ])
            elif isinstance(block, TransformerBlockNumPy):
                attn = block.attn
                pairs.extend([
                    (attn.WQ, attn.dWQ), (attn.WK, attn.dWK),
                    (attn.WV, attn.dWV), (attn.WO, attn.dWO),
                    (attn.bO, attn.dbO)
                ])
                if block.use_ln and block.ln1:
                    pairs.extend([(block.ln1.gamma, block.ln1.dgamma), (block.ln1.beta, block.ln1.dbeta)])
                if block.use_ffn:
                    ffn = block.ffn
                    pairs.extend([
                        (ffn.W1, ffn.dW1), (ffn.b1, ffn.db1),
                        (ffn.W2, ffn.dW2), (ffn.b2, ffn.db2)
                    ])
                    if block.use_ln and block.ln2:
                        pairs.extend([(block.ln2.gamma, block.ln2.dgamma), (block.ln2.beta, block.ln2.dbeta)])

        return pairs


def compute_cross_entropy_loss(logits, targets, mask):
    """
    Computes masked cross-entropy loss and gradient w.r.t logits.
    logits: (B, T, V)
    targets: (B, T)
    mask: (B, T)
    """
    probs = softmax(logits, axis=-1)
    B, T, V = logits.shape
    
    # Calculate loss
    batch_idx = np.arange(B)[:, np.newaxis]
    time_idx = np.arange(T)[np.newaxis, :]
    
    target_probs = probs[batch_idx, time_idx, targets]
    log_probs = np.log(np.maximum(target_probs, 1e-12))
    
    masked_loss = -log_probs * mask
    total_valid = np.sum(mask)
    loss = np.sum(masked_loss) / np.maximum(total_valid, 1.0)

    # Gradient w.r.t logits
    dlogits = probs.copy()
    dlogits[batch_idx, time_idx, targets] -= 1.0
    dlogits = dlogits * mask[:, :, np.newaxis] / np.maximum(total_valid, 1.0)

    return loss, dlogits, probs
