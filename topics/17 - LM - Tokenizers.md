# Tokenizers

## One-line definition
A **Tokenizer** is an algorithm that breaks raw human text into chunks (Tokens) and maps those chunks to numerical IDs (Token IDs) that the neural network can process.

## Why it exists
Neural networks can only perform mathematical operations on numbers. They cannot multiply the letter "A" by `0.5`. Text must be converted to numbers before entering the network.

## Beginner intuition
Imagine translating a book into a secret numeric code. 
First, you decide what your chunks are. Are you chunking by individual letters? Whole words? Syllables?
If you chunk by whole words, you build a dictionary:
- "The" = 0
- "Cat" = 1
- "Runs" = 2
When you receive the text "The Cat", the tokenizer outputs `[0, 1]`. 

## How it works in modern AI
Modern Large Language Models (like GPT-4) do not usually use "Whole Word" tokenization because there are too many words in the world (millions). They also do not use "Character" tokenization because it takes too many characters to convey meaning.
Instead, they use **Sub-word Tokenization** (like Byte-Pair Encoding or BPE). 
A word like "Unbelievable" might be split into `["Un", "believ", "able"]`. This allows the model to handle words it has never seen before by combining known sub-word chunks.

## Week 2 assignment connection
In our toy problem, our Tokenizer is incredibly simple. We are using **Whole Word Tokenization**. We have a Python dictionary mapping exact strings (`"Receive"`) to exact integers (`6`). 
```python
vocab = {"Order": 0, "Shipment": 1, ... "Receive": 6}
```
If we try to pass `"Received"` to our network, it will crash, because our simple tokenizer has no sub-word fallback mechanism.

## Common misunderstanding
**Misunderstanding:** The Tokenizer is part of the Neural Network.
**Correction:** The Tokenizer is completely separate from the Neural Network. It runs *before* the data goes in, and *after* the data comes out. It has no weights and does not learn during training via backpropagation; it is a fixed set of rules.

## Teach-back question
Why do modern language models prefer sub-word tokenization over whole-word tokenization?

## My Understanding
*(Learner space to explain this in their own words later)*.

## Flashcards

Is the Tokenizer trained by Backpropagation alongside the neural network? #card
No. The Tokenizer is a separate, pre-defined algorithm that converts text to numbers before the network sees it. It does not learn during the neural network training loop.

What is the main advantage of Sub-Word tokenization over Whole-Word tokenization? #card
It keeps the vocabulary size manageable while still allowing the model to process completely unknown words by breaking them down into familiar sub-word chunks (like prefixes and suffixes).
