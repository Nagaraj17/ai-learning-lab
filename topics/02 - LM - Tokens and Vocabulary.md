# Tokens and Vocabulary

## One-line definition
A Vocabulary is the master list of all allowed words, and a Token ID is the integer index of a specific word in that list.

## Why it exists
Computers can only process numbers, so we must map text strings ("Order") to a numerical lookup system.

## Beginner intuition
Think of the Vocabulary as a dictionary where every word is assigned a page number (Token ID). 

## Week 1 assignment connection
Our Vocabulary size is 10. The word "Order" is assigned the Token ID `1`, and "Shipment" is assigned `5`. We use a dictionary `state_to_id` to look these up.

## Small numerical example
Vocabulary: `["Apple", "Banana", "Cherry"]`
Token ID for "Banana": `1` (using 0-based indexing)

## Common misunderstanding
**Misunderstanding:** A Token ID is a meaningful mathematical value (e.g., `Banana = 1`, `Cherry = 2`).
**Correction:** Token IDs are *purely identifiers*. If you pass `2` into a math equation, the network thinks Cherry is twice as big as Banana. This is why we never pass Token IDs directly into a linear layer.

## What happens if removed or changed?
If the network encounters a word that is not in the Vocabulary (Out of Vocabulary), it crashes because it has no Token ID to assign to it.

## Teach-back question
If my vocabulary is `["Cat", "Dog", "Bird"]`, why shouldn't I pass the inputs `0`, `1`, and `2` directly into the neural network's `Wx + b` formula?

## Flashcards

Why shouldn't Token IDs be treated as ordinary continuous numerical features in a Linear Layer? #card
Because passing raw integers implies a false mathematical relationship (e.g., Token 4 is "twice as much" as Token 2). They are just lookup identifiers, not measurements.

Can an Embedding Layer accept Token IDs directly? #card
Yes. Embedding layers are designed specifically to use Token IDs as lookup indexes to retrieve vectors, avoiding the false mathematical relationship problem.
