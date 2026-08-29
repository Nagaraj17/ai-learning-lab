# 04 - Autoregressive Generation and Temperature

## 1. The Problem

You have successfully trained a Tiny-GPT model. The training loss is low. But a trained model is just a giant matrix of frozen numbers. If you feed it a clinical note, it won't magically spit out a fully de-identified note on its own. 
During training, the model processed the entire sequence at once (in parallel) because it already knew the right answers. But during generation, we don't have the right answers! How do we actually *use* the model to write new text?

## 2. Why We Need Something New

We need an inference loop (a `generate()` function) that repeatedly queries the model to build a sequence piece-by-piece, and we need a mathematical way (Temperature) to control how "creative" or "strict" the model is when picking the next piece.

## 3. One-Line Definition

**Autoregressive Generation** is the process of feeding a model its own generated outputs as input context to predict the next token, while **Temperature** is a scaling factor applied to the model's raw scores before calculating probabilities to control randomness.

## 4. Beginner Intuition / Mental Model

Think of the model as an autocomplete feature on your phone.
You type: "I am going to the ". 
The phone calculates probabilities for the next word: "store" (80%), "park" (15%), "moon" (5%).
You pick "store".
Now the sentence is "I am going to the store". You feed this *entire new sentence* back into the phone, and it predicts the next word (e.g. "to"). 
Autoregressive generation is just autocomplete running in a loop until it predicts a stop signal (`<EOS>`). Temperature is like a slider on the phone: low temperature always picks the top word ("store"), high temperature occasionally picks the weird word ("moon").

## 5. What Came Before → What Changes Now

* **Before (Training):** Parallel processing. The model sees the entire target sequence at once, using causal masking to prevent cheating.
* **Now (Inference):** Sequential processing. The model must construct the output one token at a time, appending it to the context window and re-running the entire forward pass.

## 6. How It Works

1. Encode the prompt.
2. Run the complete forward pass.
3. Slice the logits matrix to only look at the **final position** (the token we want to predict next).
4. Divide the logits by the Temperature.
5. Apply Softmax to convert to probabilities.
6. Select a token (Greedy = take highest probability; Sampling = pick randomly based on probabilities).
7. Append the token to the context.
8. Repeat until the model selects the `<EOS>` token.

## 7. Required Mathematics 

Before softmax:
$Adjusted Logits = \frac{Logits}{Temperature}$

Softmax exponentiates and normalizes:
$P(i) = \frac{e^{Adjusted Logit_i}}{\sum e^{Adjusted Logit_j}}$

## 8. Complete Worked Example

Logits for the next token:
`PATIENT: 3.0`
`[NAME]: 1.0`
`MRN: 0.0`

**Low Temperature (T = 0.1):**
Adjusted: `[30, 10, 0]`
Probabilities: `PATIENT` gets ~99.9%. Output is highly deterministic (Greedy).

**Moderate Temperature (T = 1.0):**
Adjusted: `[3.0, 1.0, 0.0]`
Probabilities: `PATIENT` (84%), `[NAME]` (11%), `MRN` (4%). Top token dominates but others have a chance.

**High Temperature (T = 5.0):**
Adjusted: `[0.6, 0.2, 0.0]`
Probabilities: `PATIENT` (41%), `[NAME]` (27%), `MRN` (22%). Distribution is flatter, making the model highly random/creative.

## 9. Math → Code Mapping

```python
# 1. Get the logits for the VERY LAST token in the sequence
next_token_logits = logits[:, -1, :] 

# 2. Apply temperature
next_token_logits = next_token_logits / temperature

# 3. Softmax
probs = torch.softmax(next_token_logits, dim=-1)

# 4. Greedy selection (if T is very low)
next_token = torch.argmax(probs, dim=-1) 
```

## 10. Experiments / What-If Questions

**What happens if the model predicts the wrong token early on?**
*Prediction:* Exposure Bias. If the model accidentally generates `MRN` instead of `PATIENT`, the next token it generates will be conditioned on the context `...<OUTPUT> MRN`. It will continue generating down this wrong path, causing the errors to cascade.

## 11. Common Misunderstandings

* **"Temperature 0 means greedy decoding."** -> Mathematically, dividing by zero crashes the code. We use a very low number (e.g. `0.1`) or an explicit `argmax` function for greedy decoding.
* **"Higher temperature creates knowledge."** -> No, it only increases the chance of selecting lower-ranked existing candidates (often causing hallucination).
* **"`<EOS>` is automatically added by the code."** -> No, `<EOS>` is a vocabulary token the model must learn to predict to signal it has finished.

## 12. Limitations and Trade-Offs

Autoregressive generation suffers from context window limits. If the combined prompt and generated output exceed the model's `max_seq_len`, the oldest tokens are dropped. For Tiny Mitra, dropping the original note from the context means the model will start hallucinating placeholders with no source data!

## 13. Where It Appears in the Current Assignment

You will write the `generate()` loop in your `TinyLanguageModel`. You must handle appending the token and breaking the loop when `<EOS>` is reached.

## 14. Where It Appears in Modern AI Systems

This is the exact loop used by all LLMs. Production systems use optimizations like KV Caching to avoid re-calculating the entire forward pass for old tokens, but the autoregressive loop remains identical.

## 15. Connection to the Next Concept

Now that we can generate text, how do we know if our model is actually good? We can't just use token accuracy, because missing a single name is a critical failure in de-identification. We need specialized metrics, which we cover in **Topic 5: Evaluation and Safety**.

## 16. Teach-Back and Small Application Exercise

**Exercise:** If you are building a medical de-identification system, would you want a high temperature (1.2) or a low temperature (0.1) during inference? Why?

## 17. Quick Revision Summary

Autoregressive generation repeatedly converts the final-position logits into one selected token, appending it to the context to predict the next. Temperature controls selection variability by sharpening or flattening the Softmax distribution.

## 18. My Understanding

*(Write your own intuition here. How would you explain Temperature to a teammate?)*

## 19. Flashcards

**Q:** Why does an error early in generation cause cascading failures?
**A:** Because autoregressive generation feeds the model its own outputs. A wrong output becomes flawed context for all future predictions.

**Q:** What does dividing logits by a temperature < 1.0 do to the probabilities?
**A:** It sharpens the probability distribution, pushing the highest logit closer to 100% and suppressing the others.

## 20. Sources
- AI Learning Lab - Tiny-GPT Core Concepts
