# Topic 4 — Autoregressive Generation, Softmax, and Temperature

## Learning goal

You should be able to explain every iteration of generation, calculate how temperature changes probabilities, and distinguish greedy decoding from sampling.

## 1. Training ends with a model, not a redacted sentence

Training learns parameters. To produce an output, we must run an inference loop:

1. encode the prompt;
2. run the complete model;
3. take logits from the final sequence position;
4. adjust logits using temperature;
5. convert logits to probabilities;
6. select one token;
7. append it;
8. repeat until `<EOS>` or a limit.

The prompt is:

```text
<BOS> <INPUT> PATIENT Olivia Martinez DIAGNOSIS NSCLC <OUTPUT>
```

The model is not asked to produce the entire redacted note in one forward pass. It produces only the distribution for the next token.

## 2. Why the final position is used

The model returns logits with shape:

```text
(B, T, V)
```

Each sequence position predicts the token immediately after that position. During generation, we need the token after the current complete context, so we select:

```python
logits = model(input_ids, attention_mask)[:, -1, :]
```

Shape becomes `(B,V)`.

If the final context token is `<OUTPUT>`, its logits predict the first output token. Once `PATIENT` is appended, the final-position logits predict the token after `PATIENT`.

## 3. Logits and softmax

Logits are unconstrained scores such as:

```text
PATIENT: 3.0
[NAME]:  1.0
MRN:     0.0
```

Softmax exponentiates and normalizes:

```text
P(i) = exp(logitᵢ) / Σⱼ exp(logitⱼ)
```

Approximate probabilities:

```text
exp(3)=20.09, exp(1)=2.72, exp(0)=1
sum=23.81

PATIENT ≈ 0.844
[NAME]  ≈ 0.114
MRN     ≈ 0.042
```

Softmax preserves ranking but turns score gaps into probability differences.

## 4. Temperature

Before softmax:

```text
adjusted logits = logits / temperature
```

### Low temperature: `0.1`

```text
[3.0, 1.0, 0.0] / 0.1 = [30, 10, 0]
```

The largest token receives almost all probability. Output is nearly deterministic.

### Moderate temperature: `0.7`

```text
[3.0, 1.0, 0.0] / 0.7 ≈ [4.29, 1.43, 0]
```

The top token remains dominant, but alternatives have meaningful probability.

### High temperature: `1.2`

```text
[3.0, 1.0, 0.0] / 1.2 = [2.5, 0.83, 0]
```

The distribution is flatter, so lower-ranked tokens are selected more often.

Temperature does not retrain the model and does not change logit ordering. It only changes how strongly differences affect sampling probabilities.

## 5. Greedy selection versus sampling

### Greedy

```python
next_id = argmax(probabilities)
```

Always chooses the most probable token. Running the same model and prompt gives the same result.

### Sampling

```python
next_id = multinomial(probabilities)
```

Treats probabilities like a weighted lottery. A token with probability 0.1 may be selected sometimes.

Our implementation uses greedy decoding for temperature `≤ 0.15` and sampling above it. This is an educational policy, not a universal rule.

For de-identification, variability is usually undesirable. A creative placeholder sequence can cause leakage or alter clinical facts.

## 6. One generation trace

Initial context:

```text
... <OUTPUT>
```

Iteration 1:

```text
final-position distribution → PATIENT selected
```

New context:

```text
... <OUTPUT> PATIENT
```

Iteration 2:

```text
final-position distribution → [NAME] selected
```

New context:

```text
... <OUTPUT> PATIENT [NAME]
```

The entire sequence is passed through the model again on every iteration in this simple implementation. No internal state is cached.

## 7. Why errors can compound

During training, the model normally sees the correct previous output tokens. During generation, it sees its own selected tokens.

If it should generate `PATIENT` but selects `MRN`, the next prediction is conditioned on the wrong history. One error can therefore shift the rest of the output.

This difference is often called **exposure bias**:

- training context contains gold/correct previous tokens;
- generation context contains model-produced previous tokens.

## 8. Stopping conditions

Generation stops when:

1. the model selects `<EOS>`; or
2. `max_new_tokens` is reached.

`<EOS>` is learned behaviour. The hard limit is a safety guard against endless output.

Stopping too early truncates clinical details. Stopping too late may repeat text, hallucinate, or leak information.

## 9. Context window truncation

The implementation uses:

```python
context = token_ids[-model.config.max_seq_len:]
```

If the combined prompt and generated output exceed the model’s context window, the oldest tokens are dropped.

For Tiny Mitra, that may remove the beginning of the original note. Later output tokens then lose access to important source content. Tests must include maximum-length and over-length prompts.

## 10. Why low temperature is not automatically safe

A low temperature makes behaviour repeatable, not correct. If the highest-logit token is an identifier rather than a placeholder, greedy decoding will confidently leak it every time.

Therefore:

```text
deterministic ≠ accurate ≠ safe
```

Temperature experiments reveal instability, but safety must be measured directly using leakage and preservation metrics.

## 11. Experiments

For every test prompt:

1. run temperature `0.1` once;
2. run `0.7` at least five times;
3. run `1.2` at least five times;
4. record the first temperature/run where an identifier leaks;
5. record the first clinical token changed or omitted;
6. count outputs that fail to emit `<EOS>`;
7. compare exact match and clinical preservation.

Set a random seed when you need reproducible sampling experiments.

## 12. Common misconceptions

**“Temperature 0 means greedy decoding.”**  
Mathematically, dividing by zero is invalid. Our function requires temperature greater than zero and uses 0.1 for near-deterministic decoding.

**“Higher temperature creates knowledge.”**  
No. It only increases the chance of selecting lower-ranked existing candidates.

**“The model remembers earlier forward passes.”**  
Not in this implementation. The enlarged token sequence is supplied again.

**“`<EOS>` is automatically added by the framework.”**  
No. It is a vocabulary token the model must learn to predict.

## 13. Explain without notes

1. Why do we use `[:, -1, :]`?
2. What is the difference between logits and probabilities?
3. Why does dividing by a small temperature sharpen softmax?
4. How are greedy selection and sampling different?
5. Why can one early error affect the entire continuation?
6. What happens when the context exceeds `max_seq_len`?
7. Why can a deterministic output still be unsafe?

## Key takeaway

Autoregressive generation repeatedly converts the current final-position logits into one selected token and appends it. Temperature controls selection variability, not model knowledge or correctness.
