# Weights, Bias, and Initialization

## One-line definition
Weights control the influence of inputs, Bias shifts the result up or down, and we initialize them randomly to break symmetry.

## Why it exists
Weights and Biases are the actual "knowledge" of the network. They are the dials that get tuned during training to make the predictions correct.

## Beginner intuition
- **Weights (`W`)**: Like volume knobs on a stereo. They decide how "loud" or important a specific input is.
- **Bias (`b`)**: Like the master volume dial. It shifts the baseline output up or down regardless of the inputs.
- **Initialization**: If two racers start in the exact same spot and take the exact same steps, they tie. Giving them different, random starting lines allows them to run different paths (breaking symmetry).

![Artificial Neuron Diagram](<../weekly curriculum/01-forward-pass-and-learning/images/artificial_neuron.png>)

## Week 1 assignment connection
Our network has `W1` & `b1` connecting Input to Hidden, and `W2` & `b2` connecting Hidden to Output. We use `np.random.randn()` to initialize `W` randomly, and `np.zeros()` to initialize `b` to zero.

## Small numerical example
Input `x = 2`
Weight `W = 3`
Bias `b = -1`
Output = `(x * W) + b` = `(2 * 3) - 1` = `5`

## Common misunderstanding
**Misunderstanding:** The input vector "knows where to go" through the network.
**Correction:** The input vector is completely dumb. The Weights and Biases determine exactly how the input is routed, amplified, or muted as it flows through the architecture.

**Misunderstanding:** Both weights and biases must be randomly initialized.
**Correction:** Only weights strictly need to be random to break symmetry. Biases are commonly and safely initialized to zero.

## What happens if removed or changed?
If all weights are initialized to zero, every neuron outputs the same value, receives the same gradient, and learns the exact same feature. You essentially just have one giant cloned neuron.

## Teach-back question
Why doesn't it matter if we initialize the biases to zero, as long as the weights are random?

## Flashcards

Why must Weights be randomly initialized instead of set to zero? #card
To break symmetry. If all weights are identical, every neuron calculates the same output and receives the same gradient, effectively acting as identical clones.

What is the difference in purpose between a Weight and a Bias? #card
Weights control the strength/influence of a specific input. Biases shift the entire result independently, allowing the network to adjust its baseline activation.
