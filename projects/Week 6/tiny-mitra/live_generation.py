"""Yield after each real model prediction, without changing frozen model code.

This adapter follows generate.py's inference contract. Regression checks compare
its final text, token probabilities and stopping behavior with that function.
"""
import torch
from tokenizer import CONTROL


def stream_generate(model, tokenizer, source, max_new_tokens=220, mode="greedy", temperature=0.7, seed=42):
    if mode not in ("greedy", "sample") or temperature <= 0 or max_new_tokens < 1:
        raise ValueError("Choose greedy/sample, positive temperature and positive output limit")
    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer.encode(tokenizer.prompt(source))
    if len(ids) >= model.config.context:
        raise ValueError("Prompt leaves no context for generation")
    generator = torch.Generator(device=device).manual_seed(seed)
    generated, trace, reason = [], [], "output_limit"
    # Each yield follows an actual forward pass. No saved target or renderer is
    # available here. A local generator gives the same sampling sequence as the
    # console path without altering global random state.
    for _ in range(max_new_tokens):
        if len(ids) >= model.config.context:
            reason = "context_limit"
            break
        with torch.no_grad():
            logits = model(torch.tensor([ids], device=device))[0, -1]
            probabilities = torch.softmax(logits / temperature if mode == "sample" else logits, dim=-1)
            chosen = int(logits.argmax()) if mode == "greedy" else int(torch.multinomial(probabilities, 1, generator=generator))
        trace.append({"id": chosen, "token": tokenizer.tokens[chosen], "probability": float(probabilities[chosen])})
        ids.append(chosen)
        if chosen == tokenizer.eos_id:
            reason = "eos"
        else:
            generated.append(chosen)
        yield {"text": tokenizer.decode(generated), "tokens": [dict(row) for row in trace], "termination": "generating" if reason != "eos" else reason, "mode": mode, "temperature": temperature if mode == "sample" else None, "seed": seed, "protocol_valid": reason == "eos" and not any(marker in tokenizer.decode(generated) for marker in CONTROL)}
        if reason == "eos":
            return
    text = tokenizer.decode(generated)
    yield {"text": text, "tokens": trace, "termination": reason, "mode": mode, "temperature": temperature if mode == "sample" else None, "seed": seed, "protocol_valid": False}
