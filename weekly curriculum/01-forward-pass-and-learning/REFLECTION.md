# Reflection: Week 1

## What We Built
In Week 1, we successfully constructed a neural network from scratch using only NumPy. We built a complete training loop, moving from raw words to token IDs, to one-hot vectors, through a hidden layer, and finally to a predicted probability distribution. We then calculated the cross-entropy loss, backpropagated the error, and updated the weights to improve the model.

## The Illusion of Generalization
Our network successfully achieved 100% accuracy on the PO Lifecycle sequence. However, we must reflect on *why* it succeeded. It succeeded because the dataset is perfectly static—there is only one correct next word for every current word. The network merely memorized the sequence. It did not learn "what an order is"; it just learned that `[1, 0, 0...]` maps to `[0, 1, 0...]`.

## Looking Ahead to Week 2
One-hot encoding treats every word as completely unrelated. "Picking" and "Packing" have no mathematical similarity. In Week 2, we will introduce **Embeddings**, allowing the model to group similar concepts together in mathematical space, which is the first true step toward generalization.
