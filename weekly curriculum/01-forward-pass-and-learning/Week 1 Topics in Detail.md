# PREREQUISITE KNOWLEDGE (Week 1)

This is the consolidated study guide for Week 1 (Forward Pass & Learning). It covers every required concept in dependency order. 

> [!NOTE]
> To keep this guide concise and easy to navigate, deep-dives and detailed manual exercises are linked to dedicated files. **Do not mark a concept complete until you have read its deep-dive note.**

---

## Stage 1: Machine Learning Foundations
[📖 Deep-Dive Note: 01 - ML - Foundations.md](../../topics/01%20-%20ML%20-%20Foundations.md)

**1. Machine learning**: Training a system to recognize patterns from data, rather than writing explicit "if-then" rules.
**2. Supervised learning**: A subfield of ML where the algorithm learns by being given the "correct answer" (labels) during training.
**3. Dataset**: The collection of examples used for training. In our toy problem, it's a lifecycle loop of 10 vocabulary words.
**4. Input and target/label**: The Input is the current state (e.g., "Order"). The Target is the state we want the network to predict (e.g., "Shipment").

---

## Stage 2: Representing Words
[📖 Deep-Dive Note: 02 - LM - Tokens and Vocabulary.md](../../topics/02%20-%20LM%20-%20Tokens%20and%20Vocabulary.md)
[📖 Deep-Dive Note: 03 - LM - One-Hot Encoding.md](../../topics/03%20-%20LM%20-%20One-Hot%20Encoding.md)

**5. Vocabulary**: The complete set of unique words the model is allowed to know (10 words in our assignment).
**6. Token**: A single distinct piece of text (like a word).
**7. Token ID**: A simple numerical identifier (like an index in an array). *Crucial Correction: Token IDs are just identifiers, not meaningful continuous numerical measurements. They should not be passed directly into a linear layer because that establishes false mathematical relationships between words.*
**8. One-hot encoding**: A vector of all zeros with a single `1`. It preserves token identity without creating artificial magnitude or ordering between categories. When used with a dense layer, `x @ W` acts as a selector that picks exactly one row from the weight matrix.

---

## Stage 3: Mathematics
[📖 Deep-Dive Note: 04 - MATH - Vectors Matrices and Shapes.md](../../topics/04%20-%20MATH%20-%20Vectors%20Matrices%20and%20Shapes.md)
[📖 Deep-Dive Note: 05 - MATH - Matrix Multiplication.md](../../topics/05%20-%20MATH%20-%20Matrix%20Multiplication.md)
> **Optional Deep-Dives for the Curious:**
> - [💡 05b - MATH - Tensors and Norms - Optional.md](../../topics/05b%20-%20MATH%20-%20Tensors%20and%20Norms%20-%20Optional.md) *(Why read it? Discover how AI models measure their own massive errors using L1/L2 Norms, and how data is actually shaped for GPU processing.)*
> - [💡 05c - MATH - Identity Inverse and Special Matrices - Optional.md](../../topics/05c%20-%20MATH%20-%20Identity%20Inverse%20and%20Special%20Matrices%20-%20Optional.md) *(Why read it? See how we can sometimes solve machine learning equations instantly with analytical math instead of slow guessing.)*
> - [💡 05d - MATH - Eigendecomposition - Optional.md](../../topics/05d%20-%20MATH%20-%20Eigendecomposition%20-%20Optional.md) *(Why read it? Understand the "magic vectors" that act as the skeleton of a matrix, laying the foundation for how AI compresses data.)*

**9. Scalar, vector, and matrix**: 
- **Scalar**: One single number.
- **Vector**: An ordered list of numbers (1D array).
- **Matrix**: A rectangular table of numbers (2D array).
**10. Shapes**: The dimensions. In NumPy, a shape `(n,)` is a flat 1D vector, `(1, n)` is a 2D row matrix, and `(n, 1)` is a 2D column matrix. 
**11. Dot product and matrix multiplication**: Efficiently computing many weighted combinations simultaneously.

---

## Stage 4: Neural Network Basics
[📖 Deep-Dive Note: 06 - NN - Weights Bias and Initialization.md](../../topics/06%20-%20NN%20-%20Weights%20Bias%20and%20Initialization.md)
[📖 Deep-Dive Note: 07 - NN - Hidden Layers and tanh.md](../../topics/07%20-%20NN%20-%20Hidden%20Layers%20and%20tanh.md)

**12. Weights**: Control the influence of inputs. Each output neuron gets its own weighted combination of the inputs.
**13. Bias**: Shifts the result, allowing the network to adjust the baseline activation regardless of the input. Biases are commonly initialized to zero.
**14. Random initialization**: Weights are initialized with *different random values* to break symmetry (ensuring neurons learn different features). (Advanced init methods like Xavier/He are deferred to `PARKING_LOT.md`).
**15. Hidden layer**: The architecture that defines internal connections between inputs and outputs.
**16. Linear transformation: Wx + b**: The fundamental, efficient mathematical operation mapping inputs to outputs.

---

## Stage 5: Activation & Output
[📖 Deep-Dive Note: 07 - NN - Hidden Layers and tanh.md](../../topics/07%20-%20NN%20-%20Hidden%20Layers%20and%20tanh.md)
[📖 Deep-Dive Note: 08 - NN - Logits Softmax and Argmax.md](../../topics/08%20-%20NN%20-%20Logits%20Softmax%20and%20Argmax.md)
> **Optional Deep-Dives for the Curious:**
> - [💡 07b - NN - Advanced Activation Functions - Optional.md](../../topics/07b%20-%20NN%20-%20Advanced%20Activation%20Functions%20-%20Optional.md) *(Why read it? Learn why modern models completely abandoned `tanh` in favor of `ReLU` to solve the catastrophic "Vanishing Gradient" problem.)*

**17. Activation function**: Introduced immediately after `Wx + b` to add non-linearity, allowing the model to learn complex patterns instead of just straight lines.
**18. tanh**: Bends inputs (ranging from negative infinity to positive infinity) strictly into a range between `-1` and `1`.
**19. Output layer**: The final transformation producing scores for our vocabulary.
**20. Logits**: Unrestricted raw class scores. They are *not* probabilities and do not need to be between 0 and 1.
**21. Softmax**: Converts logits into a probability distribution. Every value is between 0 and 1, and they all sum to 1. Subtracting the maximum logit improves numerical stability without changing the probabilities.

---

## Stage 6: Probability & Prediction
[📖 Deep-Dive Note: 08 - NN - Logits Softmax and Argmax.md](../../topics/08%20-%20NN%20-%20Logits%20Softmax%20and%20Argmax.md)

**22. Probability distribution**: The output of Softmax, assigning a percentage likelihood to every word in the vocabulary.
**23. Predicted-class confidence**: The highest probability from Softmax. *Note: 100% confidence does not guarantee correctness!*
**24. Probability assigned to the true class**: The probability the network gave to the *correct* label (used for Loss).
**25. Argmax**: A function that returns the *index* of the largest value. Used for prediction/reporting, but gradients do *not* flow through Argmax during training.
**26. Prediction**: The final output word.

---

## Stage 7: Learning Mechanisms
[📖 Deep-Dive Note: 09 - NN - Cross-Entropy Loss.md](../../topics/09%20-%20NN%20-%20Cross-Entropy%20Loss.md)
[📖 Deep-Dive Note: 10 - NN - Gradients and Backpropagation.md](../../topics/10%20-%20NN%20-%20Gradients%20and%20Backpropagation.md)

**27. Cross-entropy loss**: Loss exists to measure how wrong a prediction is. It looks purely at the *probability assigned to the true class* (`Loss = -log(true_class_prob)`). If the network is confidently wrong, the loss explodes.
**28. Gradient**: Tells us how the loss would change if a parameter (weight/bias) changed slightly. Sign gives direction, magnitude gives sensitivity.
**29. Backpropagation**: While values flow forward to produce a prediction, gradients flow *backward* to assign responsibility using the chain rule. It does *not* directly update weights; it just calculates the gradients.

> [!TIP]
> 🧮 **Manual Exercises**: Do not proceed until you have manually traced backpropagation:
> - [Level 1: Scalar Neuron Training Step](manual-exercises/01_SCALAR_NEURON_TRAINING_STEP.md)
> - [Level 2: Complete 2x2x2 Manual Backward Pass](manual-exercises/03_MANUAL_BACKPROPAGATION.md)

---

## Stage 8: Training Execution
[📖 Deep-Dive Note: 11 - NN - Learning Rate and Updates.md](../../topics/11%20-%20NN%20-%20Learning%20Rate%20and%20Updates.md)
[📖 Deep-Dive Note: 12 - NN - Training Loop.md](../../topics/12%20-%20NN%20-%20Training%20Loop.md)
> **Optional Deep-Dives for the Curious:**
> - [💡 11b - NN - Advanced Optimizers - Optional.md](../../topics/11b%20-%20NN%20-%20Advanced%20Optimizers%20-%20Optional.md) *(Why read it? Basic SGD is painfully slow. Discover `Adam` (the industry standard optimizer) and how it uses "Momentum" like a bowling ball to speed up learning.)*

**30. Learning rate**: Controls the size of the step we take. `0` means no learning. Too large means overshooting.
**31. Weight and bias update**: The actual change applied to the parameters (`new_weight = old_weight - (learning_rate * gradient)`). We use subtraction to ensure we step *downhill* towards zero loss.
**32. Epoch**: One complete pass of the entire dataset through the training loop.
**33. Training loop**: Forward Pass → Loss Calculation → Backpropagation → Parameter Update → Repeat.
**34. Evidence that learning occurred**: Verified by observing loss decreasing, true-class probabilities increasing, and predictions fundamentally changing from their initially random states.

---

## Stage 9: Reflection
[📖 Deep-Dive Note: 13 - NN - Memorization and Generalization.md](../../topics/13%20-%20NN%20-%20Memorization%20and%20Generalization.md)
> **Optional Deep-Dives for the Curious:**
> - [💡 13b - NN - Advanced Regularization - Optional.md](../../topics/13b%20-%20NN%20-%20Advanced%20Regularization%20-%20Optional.md) *(Why read it? If GPT-4 didn't use Regularization, it would just memorize Wikipedia word-for-word. Learn how `Dropout` forces models to actually learn instead of cheat.)*

**35. Memorization in the PO lifecycle problem**: Because our toy dataset is an arbitrary loop with no real semantic patterns, the network is fundamentally *memorizing* the answers, which is acceptable for this Week 1 exercise.
**36. What has not yet been demonstrated about generalization**: Successful training on these examples does not prove the model has "learned a concept" that it could apply to unseen data. (See [PARKING_LOT.md](PARKING_LOT.md) for deferred Generalization and Train/Test Split topics).