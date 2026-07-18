# Embeddings

## One-line definition
An **Embedding** is a dense vector of numbers that represents the "meaning" of a token based on its relationships with other tokens.

## Why it exists
One-Hot Encoding treats every word as completely independent and mathematically equidistant. If we want a neural network to understand that "Receive" and "Restock" are related concepts, we need a mathematical representation that can be "close" to each other in space.

## Beginner intuition
Imagine you want to describe a car. A One-Hot encoding is just the License Plate number — it uniquely identifies the car, but tells you nothing about it. 
An **Embedding** is a list of attributes: `[Speed, Weight, Cost, Off-road_Capability]`. 
A Ferrari might be `[0.9, 0.2, 0.9, 0.1]`. A Jeep might be `[0.3, 0.7, 0.4, 0.9]`. 
By looking at the vectors, we can immediately calculate how similar two cars are mathematically.

## How it works in AI
Instead of humans manually assigning attributes (like Speed or Cost), the neural network *learns* these numbers entirely on its own during training. It initializes the vectors with random numbers, and as it practices predicting next words, it slowly nudges the numbers. Words that appear in similar contexts (e.g., both "Receive" and "Restock" appear near "Inventory") will be nudged into similar numerical vectors.

## Week 2 assignment connection
In Week 2, we replace the One-Hot encoding step with an Embedding lookup. Instead of `[0, 0, 1, 0, 0]`, the word "Receive" becomes a learned vector like `[0.72, -0.15, 0.44]`. 

## Common misunderstanding
**Misunderstanding:** Embeddings are static definitions from a dictionary.
**Correction:** Embeddings are dynamic and learned. If you train a network exclusively on medical texts, the embedding for "Apple" will cluster near "Diet" or "Health". If you train it on technology news, the embedding for "Apple" will cluster near "Microsoft" and "iPhone". The meaning is derived *entirely* from the context in the training data.

## What happens if removed or changed?
Without embeddings, Large Language Models cannot understand semantics. A model would have to learn the rules of grammar separately for "The dog runs" and "The cat runs" because it wouldn't inherently know that "dog" and "cat" are similar entities.

## Teach-back question
Why does a neural network prefer a dense embedding vector over a sparse one-hot vector when trying to generalize its learning to unseen sentences?

## My Understanding
*(Learner space to explain this in their own words later)*.

## Flashcards

What is the fundamental flaw of One-Hot Encoding when trying to represent language? #card
One-Hot Encoding assumes every word is mathematically equidistant and completely independent from every other word. It captures zero semantic meaning or similarity.

Are Embedding attributes (like gender, royalty, or size) manually defined by programmers? #card
No. The neural network learns the values entirely on its own during training by observing which words frequently appear in similar contexts.
