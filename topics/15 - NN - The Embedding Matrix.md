# The Embedding Matrix (Lookup vs Multiplication)

## One-line definition
The **Embedding Matrix** is a standard weight matrix where each row corresponds to the learned embedding vector for a single word in the vocabulary.

## Why it exists
To store the learned representations of our vocabulary in an extremely efficient way.

## Beginner intuition
Think of the Embedding Matrix as a massive Excel spreadsheet. 
- Row 0 represents Token ID 0.
- Row 1 represents Token ID 1.
- Each column is a hidden dimension (an attribute).
When the network sees Token ID 3, it simply goes to Row 3 and pulls out those numbers.

## The Mathematical Truth: Lookup == Multiplication
In Week 1, we used One-Hot Vectors. If our vocabulary size was 4, Token ID 2 was `[0, 0, 1, 0]`. 
We multiplied this by our first weight matrix ($W1$): `x @ W1`.

Because of how matrix multiplication works, multiplying a one-hot vector by a matrix simply *copies the corresponding row* of the matrix. 

```text
[0, 0, 1, 0]  @  [[w00, w01],  =  [w20, w21]
                  [w10, w11],
                  [w20, w21],
                  [w30, w31]]
```

In Week 2, we realize that doing massive matrix multiplications where 99.9% of the input is `0` is a massive waste of computer memory and processing power. 
Instead of multiplying `x @ W1`, we simply use the Token ID as an index: `W1[token_id]`. 

This matrix $W1$ is now called the **Embedding Matrix** ($W_{embed}$).

## Week 2 assignment connection
In our code, we replace the `one_hot` function entirely. 
Old code:
`x_one_hot = one_hot(x_token_id)`
`hidden_state = x_one_hot @ W1`

New code:
`embedding = W_embed[x_token_id]`
`hidden_state = embedding` 
*(Note: We often pass the embedding directly into the next layers, effectively removing the need for a separate $W1$ transformation).*

## Common misunderstanding
**Misunderstanding:** An Embedding Layer is a complex algorithm.
**Correction:** An Embedding Layer is literally just a standard array/matrix where we use array indexing (`array[i]`) instead of matrix multiplication.

## Teach-back question
If an Embedding Matrix has a shape of `(10000, 256)`, what does `10000` represent, and what does `256` represent?

## My Understanding
*(Learner space to explain this in their own words later)*.

## Flashcards

What mathematical operation is a One-Hot Vector multiplied by a Weight Matrix exactly equivalent to? #card
It is exactly equivalent to an array index lookup (selecting a single row from the matrix).

Why do we use an Embedding Lookup instead of One-Hot matrix multiplication? #card
Because one-hot vectors are mostly zeros. Multiplying massive matrices by zeros wastes enormous amounts of memory and compute. An index lookup achieves the exact same mathematical result instantly.
