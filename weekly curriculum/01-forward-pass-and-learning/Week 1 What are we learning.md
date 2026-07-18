# Week 1 — Forward Pass and Learning Path

## Assignment Goal

Understand and implement the complete fundamental neural network learning loop:

**Forward Pass**
Input (Word)
→ Token ID
→ One-Hot Vector
→ Hidden Layer (Weights, Bias, tanh)
→ Raw Scores (Logits)
→ Softmax (Probability Distribution)
→ Prediction (Argmax)

**Learning (Backward Pass)**
→ Cross-Entropy Loss
→ Gradients (Backpropagation)
→ Learning Rate
→ Weight and Bias Update

The objective is not to copy the provided implementation.
The objective is to trace, understand, and manually calculate every transformation performed during training.

---

## Learning Rule

Prerequisites are learned in strictly defined dependency order. 

A prerequisite is complete **only** when I can:
1. Explain it in my own words without notes.
2. Complete a small numerical example.
3. Identify where it appears in the Week 1 code.
4. Predict what happens if it is removed or changed.
5. Apply or implement it without copying a completed solution.

Familiarity does not equal understanding. 
Executing a provided loop does not equal understanding the mechanism.

---

## Dependency Path

Consult the `PREREQUISITE_MAP.md` for the exact 36-step checklist of concepts that build sequentially from Machine Learning Foundations through to Backpropagation and Parameter Updates.