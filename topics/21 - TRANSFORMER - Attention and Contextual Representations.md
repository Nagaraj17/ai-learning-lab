# 21 - TRANSFORMER - Attention and Contextual Representations

## 1. The Problem
Embeddings are learned from the company a token kept *during training*. For example, the token `bank` learned its base embedding by looking at thousands of training examples like:
- *"The river bank flooded."*
- *"I deposited money at the bank."*
- *"The bank approved my loan."*

The single embedding vector for `bank` is an average, generalized representation of all those meanings. 

**The limitation:** At lookup time, static embeddings are completely context-blind to the *current* sentence. 
- In `"The river bank flooded"`, `bank` gets vector $[0.45, -0.12]$.
- In `"The bank approved my loan"`, `bank` gets the exact same vector $[0.45, -0.12]$.

## 2. Why We Need Something New
The embedding lookup itself does not inspect the surrounding tokens of the current sentence. We need the token representations in the sequence to interact with each other. We need a way for "bank" to look at "river" and say, "Ah, in this specific sentence, I should act like a water bank."

## 3. One-Line Definition
**Attention** is a mechanism that computes dynamic, context-dependent representations ($\mathbf{H}$) by taking a weighted sum of sequence token representations based on pairwise interaction scores.

## 4. Beginner Intuition / Mental Model
Imagine static embeddings as **dictionary definitions**. A dictionary entry for "bank" lists all meanings at once.
Attention is like a **smart translator**: when reading a sentence, it looks at the surrounding words ("river" or "money") and highlights only the relevant meaning, creating a customized, context-aware profile for "bank" in that specific sentence.

## 5. What Came Before → What Changes Now
- **Before (Bahdanau et al., 2014):** Attention was introduced in seq2seq RNNs to allow a decoder to focus on specific source encoder states instead of compressing an entire sentence into one bottleneck vector.
- **Now (Vaswani et al., 2017):** The Transformer architecture replaces sequence-aligned recurrence with **Self-Attention** and other components, allowing every token in a sequence to attend directly to every other token in parallel!

## 6. How It Works (Conceptually)
For the sentence `"The river bank flooded"`, to understand `"bank"`, the model needs to determine how relevant every other word is:
- `river` $\to$ highly relevant (0.70)
- `bank` $\to$ relevant (0.20)
- `flooded` $\to$ relevant (0.08)
- `The` $\to$ less relevant (0.02)

Attention conceptually does four things:
1. Calculates relevance between tokens.
2. Converts that relevance into percentage weights (like the 0.70 above).
3. Uses those weights to mix information from other tokens.
4. Produces a new **contextual representation**.

```text
New "bank" representation = 
  0.70 × (river information)
+ 0.20 × (bank information)
+ 0.08 × (flooded information)
+ 0.02 × (The information)
```

## 7. Required Mathematics
*Note: The detailed mathematics for exactly how these vectors are multiplied together to create contextual outputs ($\mathbf{H} = \mathbf{A}\mathbf{V}$) will be covered shortly. For now, focus entirely on the concept that Attention is a weighted mixing of information.*

## 8. Complete Conceptual Example
Let sequence length $T = 2$ (`["river", "bank"]`).
Suppose attention determines these mixing weights:
- `"river"` pays 90% attention to itself and 10% to `"bank"`.
- `"bank"` pays 70% attention to `"river"` and 30% to itself.

`"river"` stays mostly like itself (90% self-attention), with a tiny pull from `"bank"`.
`"bank"` is heavily influenced by `"river"` (70%), so its meaning completely shifts toward water.

> **Key Observation:** Both tokens get updated! Every output is a weighted blend of ALL tokens in the sequence.

## 10. Experiments / What-If Questions
- **What if all attention weights in a row are equal ($\frac{1}{T}$)?**
  Attention produces a uniform average of the value vectors, treating every word as equally relevant.
- **What if an attention weight is $1.0$ for self and $0.0$ for others?**
  There is no cross-token mixing and the output for that position equals its own value vector $\mathbf{v}_i$.

## 11. Common Misunderstandings
- **Misunderstanding:** Attention weights can be automatically interpreted as complete explanations of a model's reasoning.
- **Correction (Jain & Wallace, 2019):** Attention weights show correlation in feature mixing, but *attention is not explanation*. Alternative attention distributions can yield identical final model outputs.
- **Misunderstanding:** An untrained or small model will automatically produce intuitive, human-like attention weights.
- **Correction:** Attention weights are driven by gradient descent optimization; untrained models produce random attention weights.

## 12. Limitations and Trade-Offs
Computing pairwise attention weights requires an $O(T^2)$ matrix $\mathbf{A} \in \mathbb{R}^{T \times T}$. For long sequences ($T = 100,000$), storing and computing $T^2$ attention scores becomes computationally expensive.

## 13. Where It Appears in the Current Assignment
In **Week 3 Assignment**, you will verify how attention matrix $\mathbf{A} \in \mathbb{R}^{3 \times 3}$ transforms static log embeddings into contextual representations $\mathbf{H} \in \mathbb{R}^{3 \times 2}$.

## 14. Where It Appears in Modern AI Systems
Every modern Transformer block uses attention to convert static input embeddings into deep contextual representations across multiple layers.

## 15. Connection to the Next Concept
Now we understand that Attention is about figuring out "Which other tokens matter to me, and how much information should I take from them?" 
But how does the model *mathematically calculate* which tokens are relevant to each other? To do this, it gives tokens three separate roles: **Query, Key, and Value** (`22 - TRANSFORMER - Query Key and Value.md`).

## 16. Teach-Back and Small Application Exercise
If you have the sentence `"The bank approved my loan"`, which word do you think `"bank"` should pay the most attention to, and why?

## 17. Quick Revision Summary
- Base token embeddings give context-independent vectors regardless of sentence context.
- Attention computes dynamic contextual representations $\mathbf{H} = \mathbf{A} \mathbf{V}$.
- Attention weights $\mathbf{A}$ determine how much context is pulled from each word.

## 18. My Understanding
*Fill in your own notes on how Attention turns static embeddings into contextual representations.*

## 19. Flashcards
What is the core difference between a base token embedding and a contextual representation? #card
A base token embedding provides the same context-independent vector for a token regardless of context. A contextual representation dynamically adjusts a token's vector based on surrounding words in the sequence.

Can attention weights always be treated as definitive explanations of model decision making? #card
No (Jain & Wallace, 2019). Attention weights show feature mixing weights, but different attention distributions can yield identical predictions. They are not guaranteed explanation proofs.

## 20. Sources
- Bahdanau, Cho, & Bengio (2014) *"Neural Machine Translation by Jointly Learning to Align and Translate"*.
- Vaswani et al. (2017) *"Attention Is All You Need"*.
- Jain & Wallace (2019) *"Attention is not Explanation"*.
- Alammar, J. & Grootendorst, M. [Hands-On Large Language Models.md](file:///c:/Users/Nagar/source/repos/ai-learning-lab/resources/references/Hands-On%20Large%20Language%20Models.md), Chapter 3.

