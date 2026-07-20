# Logits, Softmax, and Argmax

> [!NOTE]
> This topic is based on Chapter 6.2 (Gradient-Based Learning - Output Units) of the *Deep Learning* textbook.

## Formal Definition
Any neural network used for classification needs to output a probability distribution over the discrete classes. The final linear layer outputs raw, unnormalized scores called **logits**. We use the **softmax** function to convert these logits into a valid probability distribution. Finally, we use **argmax** to find the index of the highest probability.

Formally, the softmax function for the $i$-th class is defined as:
$\text{softmax}(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$

## Component-by-Component Math Breakdown
Let's break down the formula $\text{softmax}(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$:
- **$\mathbf{z}$**: The vector of raw **logits** produced by the final layer of the network. These numbers can be anything (e.g., $100.5$, $-42.1$, $0.0$).
- **$z_i$**: The specific logit score for class $i$.
- **$\exp(z_i)$**: The mathematical exponential function ($e^{z_i}$). This serves two crucial purposes: it forces every number to be positive (since probabilities cannot be negative), and it heavily exaggerates differences (a slightly higher logit becomes a vastly higher exponential).
- **$\sum_j \exp(z_j)$**: This is the denominator. It adds up all the exponentiated values for every single class. 
- **Division**: By dividing the specific class's exponentiated score by the total sum, we guarantee that all the final outputs will add up to exactly $1.0$ (100%).

## Beginner Intuition & Contrasting Analogy
Think of this process like a chaotic talent show transitioning into an official winner announcement:
- **Logits:** The eccentric judge shouting random scores: "I give Apple 50 points! Banana -12 points!" These numbers are meaningless in isolation.
- **Softmax:** The official mediator stepping in: "Okay, let's normalize this. Apple has a 98% chance of winning, Banana has a 2% chance." Softmax forces the chaotic scores into a clean 100% pie chart.
- **Argmax:** The announcer at the very end: "The winner is Apple!" Argmax throws away the percentages and just points at the index that won.

![Softmax Visualization](<images/softmax_function.png>)

## Where is this used in AI?
*   **Next-Token Prediction in LLMs:** When ChatGPT predicts the next word, it doesn't just confidently spit out one word. The final layer produces a vector of **logits** with a size of ~100,000 (one score for every possible word in its vocabulary). It applies **Softmax** to turn those 100,000 scores into probabilities. Then, it rolls a weighted dice based on those probabilities to pick the next word!
*   **Image Classification:** If you build an AI to classify images of cats and dogs, the output layer will have 2 logits. Softmax guarantees the "Cat Probability" and "Dog Probability" perfectly sum to 1.0.

## Small Numerical Example
Let's trace a prediction for 3 classes:
- **Logits:** $\mathbf{z} = [3.0, 1.0, 0.1]$
- **Exponentials ($e^z$):** $[20.08, 2.71, 1.10]$
- **Sum of Exponentials:** $20.08 + 2.71 + 1.10 = 23.89$
- **Softmax Probabilities:** $[20.08/23.89, 2.71/23.89, 1.10/23.89] = [0.84, 0.11, 0.05]$
- **Argmax:** `0` (Because index 0 has the highest probability, $0.84$)

*(Source: Ian Goodfellow, Yoshua Bengio, and Aaron Courville - Deep Learning, Chapter 6.2)*

---

## Flashcards

Are Logits restricted to be between 0 and 1? #card
No. Logits are the raw, unconstrained output scores from the final layer. They can be any number from negative infinity to positive infinity.

Why do we use the Exponential function ($e^z$) inside Softmax instead of just dividing the logits by their sum? #card
Two reasons: First, logarithms and exponentials handle negative logits perfectly by turning them into small positive fractions (probabilities can't be negative). Second, exponentials exaggerate differences, making the "winner" stand out more clearly from the runner-up.
