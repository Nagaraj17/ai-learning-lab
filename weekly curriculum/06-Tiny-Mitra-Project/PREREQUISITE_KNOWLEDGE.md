# Tiny-GPT (Mitra) - Prerequisite Knowledge

This guide consolidates the essential concepts you must understand before building your Tiny-GPT for the Mitra De-Identification project. It follows the core dependency order of the architecture.

## 1. The Core Problem: Autoregressive Sequence Transformation
Standard classification models can flag text as containing PHI, but they cannot *generate* a new, clean version of the text. To actually output a de-identified medical note, we need a model that can read the original note and "speak" a new, transformed version.
We frame this as a next-token prediction problem by concatenating the input and output with control tokens:
`<BOS> <INPUT> PATIENT John DIAGNOSIS NSCLC <OUTPUT> PATIENT [NAME] DIAGNOSIS NSCLC <EOS>`
The model's only job is to continuously predict the next word. After the `<OUTPUT>` token, the "next words" it learns to predict just happen to be the de-identified version of the prompt!

## 2. Tokenization and Unseen Entities (OOV)
Before the model can predict words, those words must become numbers (Integer IDs).
- **The Vocabulary:** A fixed dictionary of words built *only* from the training data.
- **The `<UNK>` Token:** If the model encounters a name in the test data that it has never seen before (like `Olivia`), the tokenizer converts it to a special Unknown token (`<UNK>`). 
- **Why it matters:** This prevents the system from crashing on new names, while positional embeddings ensure that two different `<UNK>` tokens next to each other are still treated as distinct entities in space.

## 3. The Causal Decoder Block
The core engine of a GPT is the Causal Decoder Block. It mixes context from the sequence to enrich the meaning of each token.
- **Q, K, V:** The token's vector is projected into a Query (what I'm looking for), Key (what I have), and Value (what I will transfer).
- **Attention Scores:** We calculate compatibility by multiplying $Q \times K^T$.
- **The Causal Mask:** During training, we process the whole sequence at once. To prevent a token from "cheating" by looking at future tokens (which won't exist during live generation), we apply a Causal Mask. This mask overwrites the upper-triangle of the attention score matrix with negative infinity (`-inf`). When passed through Softmax, `-inf` becomes `0` probability. 

## 4. Autoregressive Generation & Temperature
During live inference, we don't have the answers. We must generate them in a loop.
1. Feed the prompt into the model.
2. Get the logits (raw scores) for the **last** token in the sequence.
3. Divide the logits by the **Temperature** (e.g., $T=0.1$). This controls randomness. A low temperature makes the highest score dominate (deterministic). A high temperature flattens the scores, making the model pick weird/creative words.
4. Convert to probabilities via Softmax and pick a token.
5. **Append** that token to the input sequence and run the *entire* new sequence back through the model.
6. Stop when the model predicts `<EOS>`.

*Warning: If the model predicts the wrong word early on, it will feed that error back into itself on the next loop, causing cascading hallucinations (Exposure Bias).*

## 5. Evaluation and Safety Metrics
Global "token accuracy" is a dangerous lie in de-identification. If your model gets 98 out of 100 tokens correct, but the 2 wrong tokens were the patient's real name, it is a catastrophic failure. You must measure targeted metrics:
- **Placeholder Recall:** Did it successfully place `[NAME]` where it was supposed to?
- **Placeholder Precision:** Did it accidentally place `[NAME]` over clinical data (Over-redaction)?
- **Direct Leakage Rate:** Scanning the raw output to see if the original name/MRN survived.
- **Generalization:** You must evaluate the model on *Unseen Entities* (names not in training) and *Unseen Templates* to prove it actually learned the rules of grammar, not just memorized the training data.

## Complete Worked Example: The Token Journey
1. **Input:** `PATIENT Olivia`
2. **Token ID:** `[45, 99]` (where 99 is `<UNK>`)
3. **Embedding:** IDs map to vectors of shape `(Batch, Time, D_Model)`
4. **Decoder Block:** The vectors pass through LayerNorm, QKV Attention, and the Feed-Forward Network. Shape remains `(B, T, D_Model)`.
5. **Output Projection:** Maps `D_Model` back to the Vocabulary Size. Shape is `(B, T, Vocab)`.
6. **Slice & Softmax:** Take the last time step `(B, Vocab)`, calculate probabilities, and select the next token ID!
