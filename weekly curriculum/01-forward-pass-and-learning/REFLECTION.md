# Reflection: Week 1

## A. What We Built

In Week 1, we built a neural network from scratch using only NumPy. The network takes a word from the PO Lifecycle vocabulary, converts it to a one-hot vector, passes it through a hidden layer with `tanh` activation, produces logits, applies Softmax to get a probability distribution, and uses Argmax to make a prediction. We then implemented cross-entropy loss, backpropagation, and gradient-descent weight updates inside a training loop.

---

## B. Evidence Table

> [!IMPORTANT]
> Fill this table in after running the notebook from a clean kernel (Restart → Run All). Do not fill in values from memory.

| Evidence | Before training | After training | Verified |
|---|---:|---:|---|
| Average loss (first epoch) | | | ☐ |
| Average loss (final epoch) | | | ☐ |
| Training accuracy | | | ☐ |
| True-class probability (example: "Order"→"Shipment") | | | ☐ |
| Parameters changed from initial values | Yes / No | | ☐ |
| Results reproducible with `np.random.seed(42)` | | | ☐ |

---

## C. The Memorization Boundary

The network can learn to correctly predict the next word in the PO Lifecycle sequence. It can do this because the sequence is a fixed, deterministic loop — there is exactly one correct next word for every current word.

This is **memorization**, not generalization.

The network learned that `[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]` (the one-hot for "Order") should map to `[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]` (the one-hot for "Shipment"). It did not learn what "ordering" or "shipment" means.

The test pairs (Cabinet → Drug, Drug → Invoice) were never in the training data. The notebook's test output shows whether the network correctly handles those pairs after training.

---

## D. Week 2 Preview

In Week 2 we will replace one-hot encoding with **embeddings** — dense, learned vectors that can place similar words near each other in mathematical space. This is a richer representation, but it does not automatically produce generalization. Whether a model generalizes must always be evaluated by testing on meaningful unseen examples, not assumed from the architecture alone.

---

## E. Learner Reflection Space

*Write your own answers below after completing the notebook.*

**What was the hardest concept to calculate manually?**


**What misconception did you correct during this week?**


**What can you now explain without looking at notes?**


**What still requires help or further study?**

