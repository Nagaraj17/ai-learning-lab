# Logits, Softmax, and Argmax

> [!NOTE]
> This topic is based on Chapter 6.2 (Gradient-Based Learning - Output Units) of the *Deep Learning* textbook (Goodfellow et al.).

## Why is this Concept Required?
In **Week 1: Build a Basic Prediction Machine**, after computing hidden activations with $\tanh$, our network must produce a final prediction over target classes (e.g., predicting the next token or classifying an output). Raw hidden activations cannot directly serve as probabilities because they are unconstrained numbers. We need a systematic pipeline:
1. Compute raw scores (**Logits**).
2. Convert raw scores into a valid probability distribution (**Softmax**).
3. Pick the winning class with highest confidence (**Argmax**).

---

## Formal Definition
Any neural network used for classification outputs a probability distribution over discrete classes. The final linear layer outputs raw, unnormalized scores called **logits** ($\mathbf{z}$). We use the **softmax** function to convert logits into a valid probability distribution where all values are positive and sum to $1.0$. Finally, we use **argmax** to find the index of the class with the highest probability.

Formally, the softmax probability for the $i$-th class is defined as:

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

---

## Component-by-Component Math Breakdown

### 1. The Softmax Formula: $P_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$

| Symbol | Name | Plain-English Meaning |
| :--- | :--- | :--- |
| $\mathbf{z}$ | **Logits Vector** | The vector of raw, unconstrained output scores from the final layer ($[z_1, z_2, \dots, z_K]$). |
| $z_i$ | **Class $i$ Logit** | The raw numerical score assigned to class $i$. Can be negative, zero, or positive ($-\infty$ to $+\infty$). |
| $K$ | **Total Number of Classes** | The total count of target output classes (e.g., $K=3$ for 3 classes). |
| $e$ | **Euler's Constant** | $\approx 2.71828$, base of the natural logarithm. |
| $e^{z_i}$ | **Exponential Score of Class $i$** | Converts raw score $z_i$ into a strictly positive value ($e^z > 0$). Exponentiation also exaggerates score differences. |
| $\sum_{j=1}^{K} e^{z_j}$ | **Sum of Exponentials (Denominator)** | Adds up the exponentiated scores of *all* $K$ classes. Serves as the normalizing constant. |
| $\text{softmax}(\mathbf{z})_i$ | **Softmax Probability $P_i$** | The normalized probability for class $i$. Guaranteed to be between $0.0$ and $1.0$, with $\sum_{i=1}^K P_i = 1.0$. |

### 2. Argmax: $\hat{y} = \arg\max_{i} (\mathbf{P})$

| Symbol | Name | Plain-English Meaning |
| :--- | :--- | :--- |
| $\mathbf{P}$ | **Probability Vector** | The array of normalized probabilities output by Softmax. |
| $\arg\max_i$ | **Argument of Maximum** | Scans the probability array $\mathbf{P}$ and returns the **index $i$** of the largest value, not the value itself. |
| $\hat{y}$ | **Predicted Class Index** | The final predicted class label (e.g., class index `0`). |

---

## Beginner Intuition & Contrasting Analogies

### Analogy: The Talent Show (Logits $\to$ Softmax $\to$ Argmax)
Imagine a panel of judges scoring contestants in a talent show:

1. **Logits (Raw Loud Scores):** 
   Judge shouts raw unorganized points: "Apple gets +3.0! Banana gets +1.0! Cherry gets +0.1!"
   - *Problem:* Points can be negative, don't add up to 100%, and can't be directly compared across different contests.
2. **Softmax (The Official Percentage Pie Chart):**
   The mediator steps in, exponentiates every score to make them positive, and divides by the total:
   - "Apple has **84%** probability of winning."
   - "Banana has **11%** probability of winning."
   - "Cherry has **5%** probability of winning."
   - *Result:* Clean, positive numbers that sum up to exactly $100\%$ ($1.0$).
3. **Argmax (The Final Winner Announcement):**
   The host points to Apple: "The winner is **Contestant #0** (Apple)!"
   - Argmax drops the percentages and returns the winning category index.

![Softmax Visualization](images/softmax_function.png)

---

## Where is this used in AI?

1. **Next-Token Prediction in LLMs (ChatGPT / Claude):**
   When an LLM predicts the next word, its final output layer generates a vector of **logits** across its entire vocabulary ($\sim 100,000$ words). **Softmax** converts these scores into a huge probability distribution. The AI can either use **Argmax** to pick the single most likely word or sample from the distribution.
2. **Multi-Class Classification:**
   In vision networks classifying images into categories (Cat, Dog, Bird), Softmax turns raw final-layer scores into class probabilities so the model can report its exact confidence level.

---

## Concrete Numerical Worked Example

Suppose our model predicts scores for $K = 3$ classes: **Class 0 (Apple)**, **Class 1 (Banana)**, and **Class 2 (Cherry)**.

1. **Step 1: Raw Logits ($\mathbf{z}$)**
   $$\mathbf{z} = [3.0, 1.0, 0.1]$$

2. **Step 2: Exponentiate Each Logit ($e^{z_i}$)**
   - $e^{3.0} \approx 20.086$
   - $e^{1.0} \approx 2.718$
   - $e^{0.1} \approx 1.105$

3. **Step 3: Sum Exponentials (Denominator)**
   $$\text{Sum} = 20.086 + 2.718 + 1.105 = 23.909$$

4. **Step 4: Compute Softmax Probabilities ($P_i = e^{z_i} / \text{Sum}$)**
   - $P_0 = 20.086 / 23.909 \approx \mathbf{0.840}$ (84.0%)
   - $P_1 = 2.718 / 23.909 \approx \mathbf{0.114}$ (11.4%)
   - $P_2 = 1.105 / 23.909 \approx \mathbf{0.046}$ (4.6%)
   
   *Check:* $0.840 + 0.114 + 0.046 = 1.000$ (100%).

5. **Step 5: Apply Argmax**
   $$\hat{y} = \arg\max([0.840, 0.114, 0.046]) = \mathbf{0} \quad \text{(Class 0: Apple)}$$

---

## Connection to Active Assignment
In **Week 1: Build a Basic Prediction Machine**, after the hidden layer $\mathbf{h} = \tanh(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$, you multiply by output weights to get raw logits $\mathbf{z} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$. You then apply Softmax to get predicted probabilities $\mathbf{P}$, which feed directly into your Loss function during training and Argmax during inference.

*(Reference: Ian Goodfellow, Yoshua Bengio, and Aaron Courville - Deep Learning, Chapter 6.2)*

---

## Numerical Stability: The Max-Subtraction Trick (`np.max(logits)`)

### 1. The Problem: Exponent Overflow
In theory, standard Softmax uses $P_i = \frac{e^{z_i}}{\sum e^{z_j}}$. However, in computer hardware, 64-bit floating-point numbers cap out around $10^{308}$. 

If a neural network produces a raw logit like $z_i = 1000$ (which happens frequently during un-tuned training or large scale LLM output layers):
- $e^{1000} \approx \text{infinity (`inf`)}$
- $\frac{\text{inf}}{\text{inf}} = \text{NaN (`Not a Number`)}$

This crashes model training completely!

### 2. The Solution: Shift Logits by Subtracting the Maximum
To prevent overflow, we subtract the maximum logit value $c = \max(\mathbf{z})$ from every logit before exponentiating:

$$\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_{j=1}^{K} e^{z_j - \max(\mathbf{z})}}$$

### 3. Mathematical Proof of Equivalence
Why are we allowed to subtract $\max(\mathbf{z})$ without altering the probabilities?

Using standard exponent rules ($e^{a-b} = e^a \cdot e^{-b}$):

$$\frac{e^{z_i - c}}{\sum_{j=1}^K e^{z_j - c}} = \frac{e^{z_i} \cdot e^{-c}}{\sum_{j=1}^K (e^{z_j} \cdot e^{-c})} = \frac{e^{z_i} \cdot e^{-c}}{e^{-c} \sum_{j=1}^K e^{z_j}} = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

The constant factor $e^{-c}$ cancels out perfectly from the numerator and denominator!

### 4. Component-by-Component Math Breakdown

| Symbol / Term | Math Expression | Plain-English Meaning |
| :--- | :--- | :--- |
| $\max(\mathbf{z})$ | $c = \text{np.max(logits)}$ | The highest numerical score among all output classes in the logit vector $\mathbf{z}$. |
| $z_i - \max(\mathbf{z})$ | `shifted_logits` | Every logit adjusted relative to the highest logit. The max logit becomes $0$, and all other logits become $\le 0$ (negative numbers). |
| $e^{z_i - \max(\mathbf{z})}$ | `exponentials` | Exponentiated shifted logits. Since the exponent is at most $0$, $e^0 = 1.0$ and $e^{\text{negative}} \in (0, 1)$. **No value can ever exceed $1.0$**, completely eliminating floating-point overflow (`inf`)! |
| $\sum e^{z_j - \max(\mathbf{z})}$ | `np.sum(exponentials)` | Sum of exponents (between $1.0$ and $K$). Guaranteed to be a safe, normal float. |

### 5. Beginner Intuition & Contrasting Analogy

> **Analogy: Sea Level vs. Mountain Peak Baseline**
> Imagine measuring mountain heights:
> - **Standard Softmax (Sea Level):** Measuring height from Earth's core produces massive numbers ($6,378,000$ meters). Raising those to powers explodes into infinity.
> - **Stable Softmax (Peak Baseline):** Pick the tallest mountain peak (e.g. Mount Everest at $8,848$m) and set it as $0$m. Every other mountain is now measured as a negative offset relative to Everest (e.g., K2 is $-237$m). 
> - **The Result:** The relative differences in height between all mountains remain **100% identical**, but the numbers are small, safe, and manageable!

![Numerical Stability Softmax](images/numerical_stability_softmax.svg)

---

## Math to Code Mapping

| Mathematical Step | Python / NumPy Line | Purpose |
| :--- | :--- | :--- |
| $c = \max(\mathbf{z})$ | `max_logit = np.max(logits)` | Find the largest logit for scaling. |
| $\mathbf{z}_{\text{shifted}} = \mathbf{z} - c$ | `shifted_logits = logits - np.max(logits)` | Subtract max logit so largest shifted logit is $0$. |
| $e^{\mathbf{z}_{\text{shifted}}}$ | `exponentials = np.exp(shifted_logits)` | Exponentiate safely without numerical overflow (`inf`). |
| $\frac{e^{\mathbf{z}_{\text{shifted}}}}{\sum e^{\mathbf{z}_{\text{shifted}}}}$ | `return exponentials / np.sum(exponentials)` | Normalize into valid probabilities (summing to $1.0$). |

---

## Where is this used in AI?

1. **Production Deep Learning Libraries:** Both **PyTorch** (`torch.nn.functional.softmax`) and **TensorFlow** (`tf.nn.softmax`) internally use this exact max-subtraction trick to ensure numerical stability during neural network training.
2. **Next-Word Generation in LLMs:** When predicting over a $100,000$-word vocabulary, logits can easily reach values like $+50$ or $+100$. Numerically stable softmax prevents `NaN` loss values when computing cross-entropy loss.

---

## Connection to Active Assignment
In **Week 1: `01_Next_Word_Predictor.ipynb`**, when implementing your softmax function, writing naive `np.exp(logits) / np.sum(np.exp(logits))` will crash or produce `NaN` if logits grow large during training. By using `shifted_logits = logits - np.max(logits)`, your forward pass remains stable across all training iterations.

*(Reference: Ian Goodfellow, Yoshua Bengio, and Aaron Courville - Deep Learning, Chapter 6.2)*

---

## Flashcards

Are Logits restricted to be between 0 and 1? #card
No. Logits are raw, unconstrained output scores from the final layer. They can be any real number from negative infinity to positive infinity.

Why do we use the Exponential function ($e^z$) inside Softmax instead of just dividing the logits by their sum? #card
Two key reasons: First, exponentiating turns any negative or positive score into a strictly positive number ($e^z > 0$), which is required since probabilities cannot be negative. Second, exponentials exaggerate score differences, making the top prediction stand out cleanly.

Why do we subtract `np.max(logits)` inside Softmax? #card
Subtracting `np.max(logits)` prevents floating-point overflow (`inf`/`NaN`) when exponentiating large logits. Because $e^{-c}$ cancels out in the numerator and denominator, the resulting probabilities are mathematically identical to standard Softmax while remaining numerically safe.

---

## My Understanding

*This section is for you to fill in your own words after studying this topic.*
- What are logits in simple terms?
- Why do we need Softmax after computing logits?
- What problem occurs when logits are very large (e.g., $1000$), and how does subtracting `np.max(logits)` solve it?
- What is the difference between Softmax output and Argmax output?

## Sources
- Goodfellow, Bengio, Courville - *Deep Learning* (Chapter 6.2)

