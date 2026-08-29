# Topic 1 — From Next-Token Prediction to De-Identification

## Learning goal

By the end of this topic, you should be able to explain why Tiny Mitra is still a next-token language model even though its visible job is to redact identifiers.

## 1. Two descriptions of the same model

Tiny Mitra can be described at two levels:

- **Application task:** accept a clinical note, remove identifying information, and retain medical information.
- **Learning objective:** predict the next token from the tokens that appear to its left.

These statements do not contradict one another. Prompt formatting converts the application task into a sequence-completion problem.

Suppose the original note is:

```text
PATIENT John Smith DIAGNOSIS NSCLC
```

The desired result is:

```text
PATIENT [NAME] DIAGNOSIS NSCLC
```

For training, we join them into one sequence:

```text
<BOS> <INPUT> PATIENT John Smith DIAGNOSIS NSCLC <OUTPUT> PATIENT [NAME] DIAGNOSIS NSCLC <EOS>
```

The model does not receive a special `redact()` operation. It sees tokens and learns that text after `<OUTPUT>` should be a transformed continuation of text after `<INPUT>`.

## 2. Why the control tokens matter

| Token | Meaning |
|---|---|
| `<BOS>` | The sequence begins here |
| `<INPUT>` | The original note follows |
| `<OUTPUT>` | Begin generating the transformed note |
| `<EOS>` | The transformed note is complete |
| `<PAD>` | Empty batch space; not real content |
| `<UNK>` | Token absent from the fitted vocabulary |

Without `<OUTPUT>`, the model has no clear boundary between source and transformed text. It may continue the note instead of starting redaction. Without `<EOS>`, it has no learned stopping signal and must rely only on a fixed token limit.

## 3. What next-token prediction means

For a sequence `x₀, x₁, ..., xₜ`, the model estimates:

```text
P(xₜ₊₁ | x₀, x₁, ..., xₜ)
```

It does not output one certain answer. It produces one score, called a **logit**, for every vocabulary token. Softmax converts those scores into probabilities.

After this context:

```text
<BOS> <INPUT> PATIENT John Smith DIAGNOSIS NSCLC <OUTPUT>
```

the model might assign:

| Candidate | Probability |
|---|---:|
| `PATIENT` | 0.82 |
| `[NAME]` | 0.07 |
| `MRN` | 0.04 |
| `NSCLC` | 0.02 |
| Remaining tokens | 0.05 |

If `PATIENT` is selected, it is appended. The enlarged sequence is run through the model again, perhaps predicting `[NAME]`. This repeated process is autoregressive generation.

## 4. Inputs and targets are shifted

```text
Full sequence: <BOS> <INPUT> PATIENT John Smith ... <OUTPUT> PATIENT [NAME] ... <EOS>
Input IDs:     <BOS> <INPUT> PATIENT John Smith ... <OUTPUT> PATIENT [NAME] ...
Targets:       <INPUT> PATIENT John Smith ... <OUTPUT> PATIENT [NAME] ... <EOS>
```

At input position `t`, the model predicts target position `t`, which is the original token at `t + 1`.

That is why the dataset code uses:

```python
input_ids = ids[:-1]
targets = ids[1:]
```

The final token cannot be an input because no following token remains to predict. The first token cannot be a target because it has no preceding context in this sequence.

## 5. Why loss is masked on the prompt

The Transformer calculates logits at every input position, but our application cares about learning the output continuation:

```text
Prompt portion: 0 0 0 0 0 0
Output portion: 1 1 1 1 1
Padding:        0 0 0
```

For each supervised position:

```text
lossₜ = -log(probability assigned to the correct next token)
```

If the correct token receives probability `0.8`, loss is about `0.223`. If it receives `0.01`, loss is about `4.605`. Confidently wrong predictions are punished much more.

The code computes each token loss before masking:

```python
token_losses = cross_entropy(..., reduction="none")
loss = (token_losses * loss_mask).sum() / loss_mask.sum()
```

Loss masking does **not** hide the prompt from attention. Generated output tokens may still read earlier prompt tokens. It only decides which prediction errors update parameters.

## 6. Training versus inference

### Training

The complete correct sequence is available. Causal masking ensures each position sees only itself and earlier positions. Predictions for all positions are calculated in parallel.

### Inference

Only this exists initially:

```text
<BOS> <INPUT> original note <OUTPUT>
```

The model predicts one token, appends it, and repeats. Training is parallel across sequence positions; generation is sequential because every selected token becomes context for the following prediction.

## 7. Why this is not token classification

| Token classification | Tiny Mitra generation |
|---|---|
| One label per input token | One vocabulary distribution per generated position |
| Output length follows input length | Output length is learned |
| Redaction applied after labels | Placeholders generated directly |
| No autoregressive decoding required | Predict and append until `<EOS>` |

A production de-identifier may prefer classification and deterministic rules. We use generation because the exercise is about understanding a decoder-only language model.

## 8. What the model must learn

It must simultaneously:

1. copy safe structural and clinical tokens;
2. replace identifying spans with placeholders;
3. preserve order;
4. start transformation after `<OUTPUT>`;
5. stop by generating `<EOS>`.

This is harder than spotting a name: every output position requires choosing what should come next.

## 9. Common misconceptions

**“The model edits input tokens in place.”**  
No. It generates another sequence after `<OUTPUT>`.

**“Loss masking means the model cannot see the prompt.”**  
No. Attention sees earlier prompt tokens; the loss mask only controls training errors.

**“The model makes one de-identification prediction.”**  
No. It makes many next-token predictions.

**“Low average loss guarantees no leakage.”**  
No. One leaked name can be clinically unacceptable despite low average loss.

## 10. Trace this yourself

Use:

```text
<BOS> <INPUT> MRN 456789 PATIENT Olivia Martinez DIAGNOSIS ASTHMA <OUTPUT>
```

Answer:

1. What should the first generated token be?
2. What is predicted after that token is appended?
3. Which source tokens should be copied?
4. Which spans should become placeholders?
5. Why is each decision still next-token prediction?
6. What happens if `<OUTPUT>` is missing?
7. What changes if prompt tokens also contribute to loss?

## Key takeaway

Tiny Mitra learns de-identification because the training format makes a redacted note the expected continuation of an original note. The transformation is produced through repeated next-token prediction.
