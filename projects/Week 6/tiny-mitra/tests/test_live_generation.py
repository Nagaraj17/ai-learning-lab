"""Live yields must agree with the frozen, benchmarked inference path."""
import pytest
import torch
from config import ModelConfig
from model import TinyGPT
from tokenizer import CharacterTokenizer
from generate import generate
from live_generation import stream_generate


@pytest.mark.parametrize("mode", ["greedy", "sample"])
@pytest.mark.parametrize("limit", [1, 5])
def test_live_tokens_match_frozen_generator(mode, limit):
    torch.manual_seed(9)
    tokenizer = CharacterTokenizer()
    model = TinyGPT(ModelConfig(architecture="transformer_2", vocab_size=len(tokenizer.tokens), d_model=8, n_heads=2, d_ff=16, context=64))
    expected = generate(model, tokenizer, "Jo", max_new_tokens=limit, mode=mode)
    frames = list(stream_generate(model, tokenizer, "Jo", max_new_tokens=limit, mode=mode))
    assert frames[-1] == expected
    # Earlier frames retain their own trace rather than sharing a growing list.
    assert len(frames[0]["tokens"]) == 1
    assert all(len(frame["tokens"]) <= limit for frame in frames)
