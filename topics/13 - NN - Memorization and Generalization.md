# Memorization and Generalization

> [!NOTE]
> This topic is based on Chapter 5.2 (Capacity, Overfitting and Underfitting) of the *Deep Learning* textbook.

## Formal Definition
The central challenge in machine learning is that our algorithm must perform well on new, previously unseen inputs — not just those on which our model was trained. The ability to perform well on previously unobserved inputs is called **generalization**.

We measure this by evaluating the model's **Test Error** (on a separate test dataset), compared to its **Training Error**. If a model drives its Training Error to $0$, but its Test Error remains high, it has simply **memorized** (overfit) the training data without learning the underlying rules.

## Component-by-Component Math Breakdown
- **Training Error ($\frac{1}{N_{\text{train}}} \sum L(y_{\text{train}}, \hat{y}_{\text{train}})$):** The average loss calculated on the exact data used to update the weights. This number will naturally go down.
- **Test Error ($\frac{1}{N_{\text{test}}} \sum L(y_{\text{test}}, \hat{y}_{\text{test}})$):** The average loss calculated on a *quarantined* set of data that the network has never been allowed to see or train on. 
- **The Gap:** The difference between the Test Error and Training Error. If this gap grows large, the model is failing to generalize.

## Beginner Intuition & Contrasting Analogy
Imagine studying for a final exam in Calculus.
- **Memorization (Overfitting):** You steal the teacher's practice worksheet and perfectly memorize the answers to all 10 questions. You score 100% on the practice sheet. But on the real exam, the teacher changes the numbers. You score a 0%. You learned the answers, not the math.
- **Generalization:** You actually study the underlying formulas (the rules of calculus). You might only score 90% on the practice sheet, but on the real exam with completely new numbers, you also score a 90%!

![Generalization vs Overfitting](<images/generalization_vs_overfitting.png>)

## Where is this used in AI?
*   **Train/Validation/Test Splits:** Every dataset in HuggingFace or Kaggle is strictly split into 3 parts. You train on 80% of the data. You use 10% (Validation) to tune the learning rate. You use the final 10% (Test) exactly once at the very end to prove to the world that your model actually works and didn't just memorize the 80%.
*   **Our Week 1 Assignment:** Our Week 1 task (predicting the PO Lifecycle loop) is purely a memorization task! Because the loop is a fixed sequence of 10 words repeating infinitely, there are no "new" or "unseen" examples. The model is just acting as a lookup table.

## Small Numerical Example
Let's look at the gap between Training and Testing Accuracy:
- **Model A:** `100%` Train Acc | `40%` Test Acc $\rightarrow$ *Severe Overfitting (Memorization).*
- **Model B:** `50%` Train Acc | `45%` Test Acc $\rightarrow$ *Underfitting (Model is too weak/dumb).*
- **Model C:** `92%` Train Acc | `89%` Test Acc $\rightarrow$ *Excellent Generalization!*

Model C is the winner, even though Model A had a "perfect" training score.

*(Source: Ian Goodfellow, Yoshua Bengio, and Aaron Courville - Deep Learning, Chapter 5.2)*

---

## Flashcards

What is the difference between Memorization and Generalization in Machine Learning? #card
Memorization is achieving high accuracy strictly on the data used for training. Generalization is the ability to achieve high accuracy on completely new, unseen data (the ultimate goal of ML).

Why does our Week 1 PO Lifecycle toy problem only teach Memorization? #card
Because the dataset is a single fixed, repeating loop of 10 words. There is no "unseen" data available to test the model with, so we only prove that it can memorize the specific loop pattern.
