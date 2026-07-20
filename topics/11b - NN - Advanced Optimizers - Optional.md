# Advanced Optimizers (Optional)

> [!NOTE]
> This topic is based on Chapter 8.3 and 8.5 (Optimization Algorithms) of the *Deep Learning* textbook.

## Formal Definition
Basic Stochastic Gradient Descent (SGD) calculates the gradient and takes a single step. However, SGD can be extremely slow to navigate long, narrow valleys in the loss landscape. Modern deep learning uses **Adaptive Optimizers** (like Adam) which incorporate **Momentum** and adaptive learning rates for every single parameter.

## Component-by-Component Math Breakdown
The **Adam** (Adaptive Moment Estimation) optimizer keeps track of two things for every weight:
1. **$m$ (First Moment / Momentum):** A running average of past gradients.
2. **$v$ (Second Moment / Velocity):** A running average of past squared gradients.

The Adam update rule is roughly:
$\mathbf{\theta}_{\text{new}} = \mathbf{\theta}_{\text{old}} - \frac{\epsilon}{\sqrt{v} + \delta} \cdot m$

- **$m$**: Instead of just using the current gradient, we use the average of the last few gradients. This smooths out wild zig-zags.
- **$\sqrt{v}$**: We divide the step size by the square root of the historical variance. If a weight has been jumping around wildly (high $v$), Adam shrinks its learning rate. If a weight has been moving slowly and consistently (low $v$), Adam boosts its learning rate.
- **$\delta$**: A microscopic number (like $10^{-8}$) added just to prevent mathematically dividing by zero.

## Beginner Intuition & Contrasting Analogy
Imagine trying to hike down a mountain into a valley.
- **SGD (The Drunk Hiker):** Takes a step, looks at the ground, turns completely, takes another step, turns completely. If the valley is shaped like a half-pipe, SGD will zig-zag violently from wall to wall, taking forever to reach the bottom.
- **Adam (The Heavy Bowling Ball):** If you roll a heavy bowling ball down the half-pipe, it has **Momentum**. Even if the slope tells it to turn sharply right, its sheer weight forces it to keep rolling mostly forward. It smooths out the zig-zags and rockets down the center of the valley! Furthermore, Adam is like a bowling ball with custom brakes on every axis, adapting to the specific terrain of the mountain.

![Adam vs SGD Contour](<images/adam_vs_sgd_contour.png>)

## Where is this used in AI?
*   **The Default Standard:** Adam is so mathematically superior that it is the default optimizer for almost every major AI model today. If you look at the PyTorch training code for Large Language Models (like LLaMA or GPT), you will see `torch.optim.AdamW` (Adam with Weight Decay) used 99% of the time. SGD is rarely used in modern production AI without heavy modifications.

## Small Numerical Example
Imagine a weight's gradient over 3 steps is: `+10, -10, +10`.
- **SGD:** Updates the weight wildly: `-10`, then `+10`, then `-10`. It achieves zero actual forward progress.
- **Momentum (Adam):** Averages the gradients over time. The average of `+10, -10, +10` is roughly `+3.3`. Adam realizes the wild swings are useless noise, ignores them, and takes a smooth, consistent step of `-3.3`.

*(Source: Ian Goodfellow, Yoshua Bengio, and Aaron Courville - Deep Learning, Chapter 8)*

---

## Flashcards

What is Momentum in the context of Neural Network Optimization? #card
Momentum is the process of keeping a running average of past gradients. Like a heavy bowling ball, it forces the optimizer to keep moving in the same general direction, preventing it from getting stuck in wild zig-zag patterns.

Why is the Adam optimizer generally preferred over basic Stochastic Gradient Descent (SGD)? #card
Because Adam adaptively scales the learning rate for *every single parameter individually* based on its historical variance, and uses momentum to smooth out the path, converging much faster than SGD.
