"""Read in order: LayerNorm -> attention -> block -> TinyGPT.

Only basic PyTorch primitives are used. Shapes: B=batch, T=time,
D=model features, H=heads, Dh=features per head, V=vocabulary size.
"""
import math
import torch
from torch import nn
from config import ModelConfig


class LayerNorm(nn.Module):
    """Normalize each token's features; batch/time positions are not mixed."""
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))
        self.offset = nn.Parameter(torch.zeros(width))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # Population variance (divide by D), not the unbiased sample estimate.
        variance = (x - mean).square().mean(dim=-1, keepdim=True)
        return (x - mean) / torch.sqrt(variance + self.eps) * self.scale + self.offset


class CausalAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_width = heads, width // heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.last_attention = None

    def forward(self, x, valid=None, gates=None, capture=False):
        batch, time, width = x.shape
        # view splits features; transpose moves H before T for per-head matmul.
        def split(projection):
            return projection(x).view(batch, time, self.heads, self.head_width).transpose(1, 2)
        q, k, v = split(self.q), split(self.k), split(self.v)  # B,H,T,Dh
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_width)  # B,H,T,T
        future = torch.ones(time, time, device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(future, float("-inf"))
        if valid is not None:
            if not bool(valid[:, 0].all()):
                raise ValueError("Right padding required; first token must be valid")
            scores = scores.masked_fill(~valid[:, None, None, :].bool(), float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # each query distributes mass over eligible keys
        self.last_attention = weights.detach().cpu() if capture else None
        mixed = weights @ v  # B,H,T,Dh
        if gates is not None:
            gate = torch.as_tensor(gates, device=x.device, dtype=x.dtype)
            if gate.shape != (self.heads,) or not bool(((gate == 0) | (gate == 1)).all()):
                raise ValueError("Provide one binary gate per head")
            # Intervention: zero a head's context BEFORE head merge and output projection.
            mixed = mixed * gate[None, :, None, None]
        merged = mixed.transpose(1, 2).contiguous().view(batch, time, width)
        return self.output(merged)


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_ffn = config.architecture != "no_ffn"
        self.residual = config.architecture != "no_residual"
        normalized = config.architecture != "no_layernorm"
        self.norm1 = LayerNorm(config.d_model) if normalized else nn.Identity()
        self.attention = CausalAttention(config.d_model, config.n_heads)
        self.norm2 = (LayerNorm(config.d_model) if normalized else nn.Identity()) if self.use_ffn else nn.Identity()
        self.ffn = nn.Sequential(nn.Linear(config.d_model, config.d_ff), nn.GELU(), nn.Linear(config.d_ff, config.d_model)) if self.use_ffn else None

    def forward(self, x, valid, gates=None, capture=False):
        update = self.attention(self.norm1(x), valid, gates, capture)
        x = x + update if self.residual else update
        if self.use_ffn:
            update = self.ffn(self.norm2(x))  # independent nonlinear transform at each position
            x = x + update if self.residual else update
        return x


class TinyGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context, config.d_model)
        architecture = config.architecture
        self.bare_attention = None
        if architecture in ("single_attention", "multi_attention"):
            self.bare_attention = CausalAttention(config.d_model, 1 if architecture == "single_attention" else config.n_heads)
        depth = 0 if architecture in ("embedding_only", "single_attention", "multi_attention") else (1 if architecture == "transformer_1" else 2)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(depth)])
        self.final_norm = LayerNorm(config.d_model) if depth and architecture != "no_layernorm" else nn.Identity()
        self.vocabulary_projection = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, ids, valid=None, head_gates=None, capture=False):
        if ids.ndim != 2 or ids.shape[1] > self.config.context:
            raise ValueError("Expected B,T IDs within the configured context")
        if valid is not None and valid.shape != ids.shape:
            raise ValueError("Padding mask shape must match IDs")
        time = ids.shape[1]
        positions = torch.arange(time, device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)  # addition, not concatenation
        gates = head_gates or {}
        if any(index < 0 or index >= max(1 if self.bare_attention else 0, len(self.blocks)) for index in gates):
            raise ValueError("Unknown attention layer in head gates")
        if self.bare_attention is not None:
            x = self.bare_attention(x, valid, gates.get(0), capture)
        for index, block in enumerate(self.blocks):
            x = block(x, valid, gates.get(index), capture)
        return self.vocabulary_projection(self.final_norm(x))  # B,T,V: raw scores, not probabilities

    def attention_matrices(self):
        modules = [self.bare_attention] if self.bare_attention is not None else [block.attention for block in self.blocks]
        return [module.last_attention for module in modules]

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
