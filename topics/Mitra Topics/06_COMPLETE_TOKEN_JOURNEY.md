# 06 - Complete Token Journey: Input to Next-Token Prediction

## 1. The Problem

We've learned about Tokenization (converting words to numbers), Embeddings (giving numbers meaning), Attention (mixing context), and the Decoder Block (processing the context). But it is easy to lose the forest for the trees. 
If we feed `PATIENT Olivia Martinez DIAGNOSIS NSCLC` into the model, how do all these separate mathematical operations string together to eventually spit out `[NAME]` as the next token? 

## 2. Why We Need Something New

We need a unified mental model that traces exactly how the shape and meaning of the data change at every step, from the moment a string enters the system to the moment a new string is generated.

## 3. One-Line Definition

The **Complete Token Journey** is the end-to-end forward pass of a causal language model, mapping discrete token IDs into dense vectors, processing them through stacked decoder blocks, and projecting them back into a vocabulary probability distribution.

## 4. Beginner Intuition / Mental Model

Imagine a factory assembly line. 
1. **Raw Materials:** Words are boxed into standard crates (Token IDs).
2. **Painting:** Each crate is painted with a specific color representing its meaning (Token Embeddings) and stamped with a serial number for its position (Positional Embeddings).
3. **The Mixer (Attention):** The crates move down a conveyor belt. A machine looks at the colors of past crates and mixes some of their paint into the current crate, enriching its color.
4. **The Oven (FFN):** The crate goes through an oven that chemically alters the paint to bring out new hues.
5. **Quality Control (Output Projection):** At the very end of the belt, a scanner looks at the final color of the *last* crate on the belt and matches it against a color catalog (Vocabulary Logits) to decide what new crate to build next.

## 5. What Came Before → What Changes Now

* **Before:** Learning isolated mathematical concepts (softmax, cross-entropy, matrices).
* **Now:** Connecting the pipeline end-to-end with concrete tensor shapes.

## 6. How It Works

Here is the exact journey of a token sequence through a Tiny-GPT.

1. **Tokenization:** Text $\rightarrow$ List of Integer IDs of shape `(Batch, Time)`
2. **Embeddings:** Integer IDs $\rightarrow$ Dense Vectors of shape `(Batch, Time, D_Model)`. Positional vectors are added here.
3. **Decoder Blocks (Repeated N times):** 
   - **Q/K/V Projection:** `(Batch, Time, D_Model)`
   - **Multi-Head Attention:** Splitting `D_Model` into `H` heads, calculating scores, masking, mixing values.
   - **Feed-Forward Network:** Expanding to `D_FF` and contracting back to `D_Model`.
4. **Vocabulary Projection:** `(Batch, Time, D_Model)` $\rightarrow$ `(Batch, Time, Vocab_Size)`.
5. **Generation:** We slice the matrix to only look at the *last* time step, yielding `(Batch, Vocab_Size)`. Softmax gives us the probability of the next token.

## 7. Master Architecture Diagram

```mermaid
flowchart TD
    %% Input Layer
    Input["Prompt: PATIENT Olivia"] --> Tokens["Tokenizer: [24, 89]"]
    
    %% Embedding Layer
    subgraph Embedding Layer
        Tokens --> TE["Token Embeddings (B, T, D)"]
        Tokens --> PE["Positional Embeddings (T, D)"]
        TE --> AddE((+))
        PE --> AddE
    end
    
    %% Transformer Blocks
    subgraph Transformer Body
        AddE --> DB1["Decoder Block 1"]
        DB1 --> DB2["Decoder Block 2"]
        DB2 --> DB3["Decoder Block ... N"]
    end
    
    %% Output Layer
    subgraph Projection & Prediction
        DB3 --> LNFinal["Final LayerNorm"]
        LNFinal --> VocabProj["Vocabulary Projection (B, T, Vocab)"]
        VocabProj --> Slice["Slice Last Logit (B, Vocab)"]
        Slice --> Temp["Apply Temperature"]
        Temp --> Softmax["Softmax Probabilities"]
        Softmax --> Pick["Sample Next Token"]
    end
    
    Pick -->|"[102]"| OutputText["Generated Text: PATIENT Olivia [NAME]"]
    
    %% The Autoregressive Loop
    OutputText -->|Append and Loop!| Input
    
    style Input fill:#ffe0b2,stroke:#f57c00
    style OutputText fill:#c8e6c9,stroke:#388e3c
    style Embedding Layer fill:#e1f5fe,stroke:#0288d1
    style Transformer Body fill:#f3e5f5,stroke:#7b1fa2
    style Projection & Prediction fill:#fff3e0,stroke:#e65100
```

## 8. Required Mathematics (Tensor Shapes)

Assume `Batch = 1`, `Time = 8`, `D_Model = 24`, `Heads = 4`, `Vocab = 80`.

| Stage | Shape |
|---|---|
| Token IDs | `(1, 8)` |
| Combined Embeddings | `(1, 8, 24)` |
| Q, K, V (after split) | `(1, 4, 8, 6)` |
| Attention Weights | `(1, 4, 8, 8)` |
| Decoder Block Output | `(1, 8, 24)` |
| All-position Logits | `(1, 8, 80)` |
| Final-position Logits | `(1, 80)` |

## 9. Complete Worked Example

Let's trace the sequence `<BOS> <INPUT> PATIENT Olivia Martinez DIAGNOSIS NSCLC <OUTPUT>`. (8 tokens).
The vocabulary projection outputs an `(1, 8, 80)` tensor. We only care about the logits at index `-1` (the `<OUTPUT>` token's position).
The shape becomes `(1, 80)`.
We apply Temperature and Softmax, and token ID `45` (which maps to `PATIENT`) has the highest probability.
We append `PATIENT` to the original sequence, making it 9 tokens long, and send the *entire 9-token sequence* through the factory again.

## 10. Math → Code Mapping

```python
# The complete forward pass
x = self.token_embedding(input_ids) + self.position_embedding(positions)
for block in self.blocks:
    x = block(x, padding_mask)
x = self.final_norm(x)
logits = self.vocabulary_projection(x)

# Generation step
next_token_logits = logits[:, -1, :]
probs = torch.softmax(next_token_logits, dim=-1)
next_token = torch.argmax(probs, dim=-1)
```

## 11. Experiments / What-If Questions

**What if we slice `logits[:, 0, :]` instead of `logits[:, -1, :]` during generation?**
*Prediction:* We would be looking at the predictions made by the very first token (`<BOS>`). The model would predict `<INPUT>` over and over again, completely ignoring the rest of the prompt!

## 12. Common Misunderstandings

* **"The model remembers earlier forward passes."** -> No. In our simple implementation, the *entire* enlarged token sequence is supplied again from scratch on every iteration. (Production models use "KV Caching" to save time, but the math is identical).
* **"Why is concatenating four 6-dimensional heads equal to 24?"** -> Because $4 \times 6 = 24$. We just split the `D_Model` feature dimension across the heads so they can look for different things in parallel.

## 13. Limitations and Trade-Offs

Because we pass the entire sequence through the model on every single token generation step, the computational cost grows quadratically as the sequence gets longer. Generating a 1000-token note takes significantly longer than a 10-token note.

## 14. Where It Appears in the Current Assignment

This is the `forward()` method of your `TinyLanguageModel` class and the `generate()` loop that runs it.

## 15. Where It Appears in Modern AI Systems

This exact token journey is what happens inside a GPU when you watch ChatGPT type out an answer word by word. The shapes are just much larger (e.g. `D_Model = 4096`).

## 16. Connection to the Next Concept

You now understand the complete architecture of a Tiny-GPT. The next step is evaluating it. How do we prove it actually learned to de-identify, rather than just memorizing our training data? We explore this in the Evaluation metrics section.

## 17. Teach-Back and Small Application Exercise

**Exercise:** If you change the number of Attention Heads from 4 to 8 (keeping `D_Model` at 24), what is the new shape of the Q/K/V tensors after splitting into heads? Does the final output shape of the Decoder Block change?

## 18. Quick Revision Summary

At every generation step, one sequence becomes `(B,T,D)` contextual representations, then `(B,T,V)` vocabulary logits, and finally one selected next token. Appending that token starts the same journey again.

## 19. My Understanding

*(Write your own intuition here. Try to draw the tensor shapes changing on a piece of paper.)*

## 20. Flashcards

**Q:** Why do we only use the last position's logits during generation?
**A:** Because we want to predict the token that comes *after* our current context, which is what the final token's representation has been trained to predict.

**Q:** What is the shape of the tensor that goes into the Feed-Forward Network?
**A:** `(Batch, Time, D_Model)`

## 21. Sources
- AI Learning Lab - Tiny-GPT Core Concepts
- *Build a Large Language Model (From Scratch)* by Sebastian Raschka (End-to-end architecture flowcharts)
