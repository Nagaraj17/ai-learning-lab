"""Actual autoregressive inference. No gold targets or replacement rules here."""
import torch
from tokenizer import CharacterTokenizer, CONTROL


@torch.no_grad()
def generate(model, tokenizer, source, max_new_tokens=220, mode="greedy", temperature=0.7, seed=42, head_gates=None):
    if mode not in ("greedy", "sample") or temperature <= 0 or max_new_tokens < 1:
        raise ValueError("Choose greedy/sample, positive temperature and positive output limit")
    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer.encode(tokenizer.prompt(source))
    if len(ids) >= model.config.context:
        raise ValueError("Prompt leaves no context for generation")
    generator = torch.Generator(device=device).manual_seed(seed)
    generated, trace, reason = [], [], "output_limit"
    for _ in range(max_new_tokens):
        if len(ids) >= model.config.context:
            reason = "context_limit"
            break
        logits = model(torch.tensor([ids], device=device), head_gates=head_gates)[0, -1]
        probabilities = torch.softmax(logits / temperature if mode == "sample" else logits, dim=-1)
        chosen = int(logits.argmax()) if mode == "greedy" else int(torch.multinomial(probabilities, 1, generator=generator))
        trace.append({"id": chosen, "token": tokenizer.tokens[chosen], "probability": float(probabilities[chosen])})
        ids.append(chosen)
        if chosen == tokenizer.eos_id:
            reason = "eos"
            break
        generated.append(chosen)
    text = tokenizer.decode(generated)
    return {"text": text, "tokens": trace, "termination": reason, "mode": mode, "temperature": temperature if mode == "sample" else None, "seed": seed, "protocol_valid": reason == "eos" and not any(marker in text for marker in CONTROL)}


@torch.no_grad()
def inspect_prediction(model, tokenizer, source, output_prefix="", layer=0, head=0):
    """Inspect exactly the final query predicting the next token after prefix.

    Supplying a prefix is conditional/teacher-forced inspection; it is not
    evidence that the model freely generated that prefix.
    """
    model.eval()
    text = tokenizer.prompt(source) + output_prefix
    ids, offsets = tokenizer.encode_with_offsets(text)
    if len(ids) > model.config.context:
        raise ValueError("Inspection exceeds context")
    logits = model(torch.tensor([ids], device=next(model.parameters()).device), capture=True)[0, -1]
    matrices = model.attention_matrices()
    if not matrices:
        raise ValueError("Embedding-only model has no attention to inspect")
    if not 0 <= layer < len(matrices) or not 0 <= head < matrices[layer].shape[1]:
        raise ValueError("Layer/head outside this architecture")
    row = matrices[layer][0, head, -1].tolist()
    return {"input_tokens": [tokenizer.tokens[index] for index in ids], "offsets": offsets, "attention": row, "next_token": tokenizer.tokens[int(logits.argmax())], "layer": layer, "head": head, "conditional_prefix": output_prefix, "query_index": len(ids) - 1}
