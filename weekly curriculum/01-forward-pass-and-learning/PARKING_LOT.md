# Concept Parking Lot

This document stores advanced topics and concepts that naturally come up during the Week 1 assignment (Forward Pass and Learning), but have been intentionally deferred to later weeks to keep the curriculum focused on core understanding.

## Train / Test Split
- **Why it came up:** Discussing how to evaluate a model's true performance.
- **Intuition:** Splitting data ensures the model learns the underlying pattern rather than just memorizing the exact answers it was given during training.
- **Why deferred:** The Week 1 toy problem (PO Lifecycle) is specifically designed to teach memorization. Proper evaluation techniques will be covered when we deal with real patterns.
- **Suggested week:** Week 3 (Generalization & Evaluation).

## Generalization
- **Why it came up:** Discussing whether the model actually "understands" the problem.
- **Intuition:** A model generalizes when it can correctly predict outcomes for data it has never seen before.
- **Why deferred:** Week 1 focuses purely on the mechanics of getting numbers to flow through the network and update the weights.
- **Suggested week:** Week 3 (Generalization & Evaluation).

## Regularization
- **Why it came up:** Discussing how to prevent the model from overfitting (memorizing too perfectly).
- **Intuition:** Adding a penalty to the loss function that discourages the network from relying too heavily on any single weight.
- **Why deferred:** Requires a deep understanding of standard loss functions and generalization first.
- **Suggested week:** Week 4 (Optimization & Regularization).

## Bias-Variance Trade-off
- **Why it came up:** Discussing model performance and why some models fail in the real world.
- **Intuition:** Finding the sweet spot between a model that is too simple (High Bias) and a model that memorizes noise (High Variance).
- **Why deferred:** Highly theoretical concept that is easier to grasp once we have a dataset capable of overfitting.
- **Suggested week:** Week 4.

## Hyperparameter Tuning
- **Why it came up:** Trying to decide the best values for Learning Rate and Epochs.
- **Intuition:** Experimenting with the configuration dials (hyperparameters) of the network to find the combination that trains the fastest and most accurately.
- **Why deferred:** First, we must understand exactly what the learning rate does mathematically.
- **Suggested week:** Week 4.

## Xavier / He Initialization
- **Why it came up:** Discussing why standard Gaussian initialization can cause problems in deep networks.
- **Intuition:** Smart random initialization that scales the random numbers based on the number of inputs/outputs to prevent signals from exploding or vanishing.
- **Why deferred:** Our Week 1 network is only 2 layers deep, so standard initialization works perfectly.
- **Suggested week:** Week 5 (Deep Architectures).

## Vanishing / Exploding Gradients
- **Why it came up:** Discussing the limits of deep neural networks and specific activation functions.
- **Intuition:** When multiplying many small gradients together during backpropagation, the signal can shrink to zero (vanish) or grow to infinity (explode).
- **Why deferred:** Requires a solid grasp of the chain rule and deep networks.
- **Suggested week:** Week 5.

## Adam Optimizer
- **Why it came up:** Looking for faster ways to train than basic Stochastic Gradient Descent (SGD).
- **Intuition:** A smart optimizer that gives every individual weight its own dynamic learning rate based on momentum.
- **Why deferred:** Basic SGD must be manually calculated and understood before introducing momentum algorithms.
- **Suggested week:** Week 4.

## Mini-batches
- **Why it came up:** Discussing how to train on large datasets efficiently.
- **Intuition:** Processing a small chunk of the dataset (e.g., 32 examples) at a time, averaging their gradients, and taking one step, balancing speed and stability.
- **Why deferred:** In Week 1, we update weights after every single example (batch size = 1) to make the manual calculations traceable.
- **Suggested week:** Week 2 or 3.

## Embeddings
- **Why it came up:** Realizing One-Hot encoding treats every word as equally distant and doesn't capture meaning.
- **Intuition:** Learning dense vectors where similar words (like "Receive" and "Restock") are grouped closely together in mathematical space.
- **Why deferred:** This is the core focus of Week 2!
- **Suggested week:** Week 2 (Embeddings).

## Model Calibration
- **Why it came up:** Discussing whether a 99% Softmax probability actually means the model is 99% likely to be right.
- **Intuition:** Ensuring the model's confidence accurately reflects real-world probability.
- **Why deferred:** Advanced evaluation metric.
- **Suggested week:** Week 6+.
