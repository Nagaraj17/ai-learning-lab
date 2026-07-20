# Learning Rate and Updates

> [!NOTE]
> This topic is based on Chapter 4.3 (Gradient-Based Optimization) of the *Deep Learning* textbook.

## Formal Definition
Once backpropagation computes the gradient of the loss with respect to the parameters, the **Optimizer** must use this gradient to update the weights. The simplest optimization algorithm is **Stochastic Gradient Descent (SGD)**. 

The update rule scales the gradient by a hyperparameter called the **learning rate** (often denoted by $\epsilon$ or $\eta$), which determines the step size.

## Component-by-Component Math Breakdown
The mathematical formula for a Gradient Descent update is:
$\mathbf{\theta}_{\text{new}} = \mathbf{\theta}_{\text{old}} - \epsilon \mathbf{g}$

- **$\mathbf{\theta}$ (Theta)**: A mathematical placeholder for any parameter in the network (e.g., a Weight matrix $\mathbf{W}$ or a Bias vector $\mathbf{b}$).
- **$\mathbf{g}$**: The gradient we just calculated during Backpropagation ($\frac{\partial L}{\partial \theta}$). It points in the direction of the steepest *ascent* (uphill).
- **$\epsilon$ (Epsilon)**: The **Learning Rate**. This is a small positive number (like $0.01$) that determines how large of a step we take.
- **$-$ (Subtraction)**: Because the gradient $\mathbf{g}$ always points uphill towards maximum loss, we must *subtract* it to step downhill towards minimum loss.

## Beginner Intuition & Contrasting Analogy
Imagine you are walking down a mountain blindfolded, trying to find the lowest valley (where Loss = 0).
- **The Gradient** is your foot feeling the slope of the ground. It tells you which way is downhill.
- **The Learning Rate** is the size of your footstep.

If your learning rate is perfectly tuned, you take smooth strides to the bottom. 
If your learning rate is too small, you take microscopic baby steps. It will take you billions of years to reach the bottom.
If your learning rate is massive, you take a giant leap. You might overshoot the valley completely, land halfway up the opposite mountain, and bounce violently into outer space!

![Learning Rate Comparison](<images/learning_rate_comparison.png>)

## Where is this used in AI?
*   **Learning Rate Schedulers:** In modern AI training (like training LLaMA or GPT-4), we don't use a fixed learning rate. We use a "Scheduler" that starts with a large learning rate to quickly bound down the mountain, and then slowly decays the learning rate to tiny baby steps as we get close to the bottom, allowing the model to perfectly settle into the absolute lowest point of the valley without bouncing out.
*   **Hyperparameter Tuning:** The learning rate is often considered the single most important hyperparameter to tune. If an AI engineer's model isn't learning, the very first thing they check is the learning rate.

## Small Numerical Example
Let's see a single weight update in action:
- **Old weight:** $0.5$
- **Gradient ($\mathbf{g}$):** $2.0$ (Positive slope, meaning increasing the weight increases the loss)
- **Learning Rate ($\epsilon$):** $0.1$

Calculate the step size: $\epsilon \times \mathbf{g} = 0.1 \times 2.0 = 0.2$
Update the weight: $\mathbf{\theta}_{\text{new}} = 0.5 - 0.2 = \mathbf{0.3}$

The weight successfully decreased to step downhill!

*(Source: Ian Goodfellow, Yoshua Bengio, and Aaron Courville - Deep Learning, Chapter 4.3)*

---

## Flashcards

What happens if you set the Learning Rate to 0.0? #card
The parameters never update (`old_weight - 0 * gradient = old_weight`). The network does not learn at all.

Why do we use subtraction in the weight update formula (`W = W - lr * gradient`)? #card
Because the gradient mathematically points uphill (towards maximum loss). We want to minimize the loss, so we subtract the gradient to step downhill into the valley.
