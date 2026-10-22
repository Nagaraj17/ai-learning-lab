"""Serializable choices; architecture settings are different from learned weights."""
from dataclasses import asdict, dataclass

ARCHITECTURES = (
    "embedding_only", "single_attention", "multi_attention", "transformer_1",
    "transformer_2", "no_ffn", "no_layernorm", "no_residual",
)
ALIASES = {"A": "embedding_only", "B": "single_attention", "C": "transformer_2"}


@dataclass
class ModelConfig:
    vocab_size: int
    architecture: str = "transformer_2"
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 192
    context: int = 512
    pad_id: int = 0

    def __post_init__(self):
        self.architecture = ALIASES.get(self.architecture, self.architecture)
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        if min(self.vocab_size, self.d_model, self.n_heads, self.d_ff, self.context) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must divide evenly among heads")

    def to_dict(self):
        return asdict(self)
