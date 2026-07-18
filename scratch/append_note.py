import os

file_path = r"c:\Users\Nagar\source\repos\ai-learning-lab\weekly curriculum\01-forward-pass-and-learning\PREREQUISITE_KNOWLEDGE.md"

content = """
---

## 5.8 Vanishing and Exploding Gradients

**Definition.**
The "Gradient" is the learning signal that tells a neural network how to update its weights. 
- **Vanishing Gradients** occur when this signal gets smaller and smaller as it travels backwards through the network, eventually shrinking to exactly zero. When it hits zero, the network completely stops learning.
- **Exploding Gradients** occur when this signal gets larger and larger, eventually growing to infinity (or `NaN` - Not a Number). When this happens, the network's math breaks and it becomes completely unstable.

**Why it is needed.**
Understanding this phenomenon is required to understand why we carefully initialize our weights as small decimals (like multiplying by `0.1`) and why we choose specific activation functions like ReLU or Tanh. Almost all modern deep learning architecture decisions exist specifically to prevent gradients from vanishing or exploding!

**Intuition (The Telephone Game).**
Imagine a line of 100 people playing the Telephone Game. 
- **Vanishing**: If every person whispers just 10% quieter than the person before them, by the 20th person, it is absolute silence. The message is completely lost.
- **Exploding**: If every person shouts 10% louder than the person before them, by the 20th person, the volume is so deafening that the eardrums burst.

**Real-world Example (Compound Interest).**
If you multiply fractions, they shrink exponentially: $0.5 \\times 0.5 \\times 0.5 = 0.125$.
If you multiply whole numbers, they grow exponentially: $2.0 \\times 2.0 \\times 2.0 = 8.0$.
Because a Neural Network is essentially a long chain of multiplications, a chain of numbers slightly less than 1 will vanish to 0, and a chain of numbers slightly greater than 1 will explode to infinity.

**Small Numerical Example (The Tanh Problem).**
In our forward pass, we use the `Tanh` activation function. `Tanh` squashes numbers between `-1` and `1`. 
If our initial weights are large (say, 5.0 instead of 0.5), the math flowing into the Tanh function will be a large number (like `10.0`). 
The `Tanh` of `10.0` is `0.999999995`. 
If the math wants to adjust the weight to make it `11.0`, the `Tanh` of `11.0` is `0.999999999`. 
The difference (the gradient) is `0.000000004`. It is so vanishingly small that the network cannot use it to learn. It is stuck!

**Connection to the assignment.**
In the `01_Next_Word_Predictor.ipynb` notebook, we explicitly initialized our weights as small decimals:
`W1 = np.random.randn(vocabulary_size, hidden_size) * 0.1`

We multiplied by `0.1` specifically to prevent the inputs to the `Tanh` function from becoming too large. By keeping the numbers small and close to zero, we keep the network in the "steep" part of the Tanh curve, guaranteeing that the learning signal will not vanish when we train it later!

**My Understanding**
(Learner space to explain this in their own words later).

"""

with open(file_path, "a", encoding="utf-8") as f:
    f.write(content)

print("Appended successfully.")
