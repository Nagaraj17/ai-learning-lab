# Prerequisite Knowledge — Forward Pass Assignment

> **Purpose**
>
> This is a self-contained study guide. It covers every concept you need
> before implementing the forward-pass assignment. No external course or
> textbook is required.
>
> Read it from top to bottom. Each section builds on the one before it.
> By the end you will be able to trace an input word through an entire
> neural network and explain every computation that happens along the way.

---

# 1. How to Use This Guide

1. Read each section in order — concepts are arranged by dependency.
2. Pay attention to the **Connection to the assignment** paragraph in
   every concept — it tells you exactly where and why the idea appears
   in the code you will implement.
3. Section 8 contains one complete worked numerical example. After
   reading the theory, work through that example by hand to confirm
   your understanding.
4. Section 9 is a glossary for quick revision.
5. Section 10 contains readiness questions. If you can answer them
   comfortably, you are ready to begin the assignment.

---

# 2. Machine-Learning Foundations

## 2.1 Machine Learning

**Definition.**
Machine learning is a way of building systems that learn patterns from
data instead of following rules written by a programmer.

**Why it is needed.**
Some problems have so many possible inputs and edge cases that writing
explicit rules becomes impractical. Consider recognising a cat in a
photograph: no human can write enough `if/else` statements to cover
every possible arrangement of pixels. Machine learning lets the computer
discover those rules by studying thousands of examples.

**Intuition.**
Think of a junior developer joining your team. Instead of reading a
1,000-page coding standard, they study 500 pull-request reviews and
gradually learn what "good code" looks like. They are learning patterns
from examples, not from explicit rules. Machine learning works the same
way — but with numbers.

**Example.**
A spam filter trained on 10,000 emails tagged as spam or not-spam. After
studying the examples, the system can classify a new email it has never
seen before.

**Connection to the assignment.**
The forward-pass code is the first half of a machine-learning system. It
takes an input and produces a prediction. Right now the system has not
learned anything, so its prediction is essentially random. Later
assignments will add the "learning from data" part.

**Common misunderstanding.**
"Machine learning replaces all programming." It does not. Machine
learning replaces the *rule-discovery* step. You still write code to
load data, build the network, run training, and deploy the result.

---

## 2.2 Traditional Programming vs Machine Learning

**Definition.**
Traditional programming and machine learning are two different
strategies for solving a problem. The difference lies in *who writes the
rules*.

**Traditional programming formula:**

```
Input Data + Rules (written by engineer) → Output
```

**Machine learning formula:**

```
Input Data + Expected Output (examples) → Rules (learned by computer)
```

<img src="./images/traditional_vs_ml.png" alt="Traditional Programming vs Machine Learning Diagram" width="600" />

**Why it is needed.**
Understanding this distinction is the single most important mental-model
shift when moving from software engineering to AI engineering. Without
it, the rest of the journey will feel arbitrary.

**Intuition (C# analogy).**
In traditional programming you write a method:

```csharp
bool IsSpam(Email email)
{
    if (email.Subject.Contains("Free Money")) return true;
    if (email.SenderReputation < 0.3)        return true;
    return false;
}
```

You, the engineer, wrote every rule. In machine learning, you instead
provide labelled examples:

```
Email A → Spam
Email B → Not Spam
Email C → Spam
...
```

The algorithm processes these examples and produces a trained model.
That model is the equivalent of a compiled DLL containing thousands of
micro-rules the computer discovered on its own.

**Example.**
Predicting house prices. Traditional: `price = sqft * 150`. Machine
learning: give the system 10,000 sales records and let it discover that
`price = sqft * 153.2 + bedrooms * 8,400 − age * 1,200 + 12,000`.

**Connection to the assignment.**
The forward-pass code you will study *is* the learned model (or rather,
the skeleton of one). It takes an input and produces an output using
mathematical operations — no `if/else` anywhere.

**Common misunderstanding.**
"ML is always better than traditional programming." It is not. If you
can express a rule precisely (e.g., "reject all orders over ₹10 lakh
without manager approval"), traditional code is simpler, faster, and
easier to debug.

---

## 2.3 Supervised Learning

**Definition.**
Supervised learning is a type of machine learning where the system
learns from input–output pairs that a human has labelled in advance. The
word "supervised" means a teacher (the labelled data) is guiding the
learning.

**Why it is needed.**
Without labelled examples, the system has no way of knowing what the
correct answer should be. Supervised learning gives the system a
"cheat sheet" so it can compare its own guesses to the correct answers
and improve.

**Intuition.**
Imagine a student preparing for an exam using past papers *with answer
keys*. Each question (input) has a correct answer (expected output). The
student practises, checks, corrects mistakes, and gradually improves.
That is supervised learning.

**Example.**
| Input word | Expected output word |
|------------|----------------------|
| "the"      | "cat"                |
| "cat"      | "sat"                |
| "sat"      | "on"                 |

Given the word "the", the correct next word is "cat". The system tries
to guess, compares its guess to "cat", and adjusts.

**Connection to the assignment.**
The forward-pass assignment uses a small next-word-prediction task.
Each training pair is `(input word, expected next word)`. The network
will receive "the" and try to predict "cat". This is supervised learning
because we know the correct next word.

**Common misunderstanding.**
"Supervised means a human watches the computer in real time." No. The
"supervision" is the *labels* in the data, not a human sitting at the
keyboard.

---

## 2.4 Dataset

**Definition.**
A dataset is a structured collection of examples used to train or
evaluate a machine-learning model. Each example is one input–output
pair.

**Why it is needed.**
Without data there is nothing for the model to learn from. The dataset
is the "textbook" the model studies.

**Intuition.**
Think of a SQL table. Each row is one example. The columns are the
properties (features) the model will use, plus one column for the
correct answer (label).

**Example.**

| Row | Input word | Expected next word |
|-----|------------|--------------------|
| 1   | "the"      | "cat"              |
| 2   | "cat"      | "sat"              |
| 3   | "sat"      | "on"               |

This tiny dataset has three examples.

**Connection to the assignment.**
The forward-pass code defines a small vocabulary and a set of
word-pairs. That set of word-pairs is the dataset.

**Common misunderstanding.**
"More data is always better." Not necessarily. Bad data (incorrect
labels, duplicates, biased samples) can make a model worse. Quality
matters as much as quantity.

> A dataset is a collection of examples. In supervised learning, each example normally contains an input and a target label. Unsupervised datasets may contain only inputs.
---

## 2.5 Input and Output

**Definition.**
The *input* is the data the model receives. The *output* is the data
the model produces. In supervised learning, the expected output is
already known during training so the model can compare and learn.

**Why it is needed.**
Every machine-learning system has a clearly defined input and output.
Without knowing what goes in and what comes out, you cannot design or
debug the system.

**Intuition.**
Think of a C# method signature:

```csharp
string PredictNextWord(string currentWord)
```

`currentWord` is the input. The returned string is the output.

**Example.**
Input: the word `"the"`. Output: the word `"cat"`.

**Connection to the assignment.**
In the forward pass, the input is a single word from the vocabulary. The
output is the network's predicted next word. The prediction may be wrong
because the network has not learned yet.

**Common misunderstanding.**
"The output is always a single number." Not in this assignment. The
network produces a score for *every* word in the vocabulary, then picks
the word with the highest score.

---

## 2.6 Features and Labels

**Definition.**
*Features* are the input properties the model uses to make predictions.
*Labels* are the correct answers provided in the training data.

**Why it is needed.**
The split between features and labels defines what the model should pay
attention to (features) and what it should try to predict (labels).

**Intuition (database analogy).**
In a SQL table of house sales:

```
SELECT sqft, bedrooms, age  -- features
FROM   houses
```

The `price` column is the label — the value we want to predict.

In our assignment the feature is the input word and the label is the
expected next word.

**Example.**

| Feature (input word) | Label (expected next word) |
|----------------------|----------------------------|
| "the"                | "cat"                      |
| "cat"                | "sat"                      |

**Connection to the assignment.**
Each word-pair in the code defines one feature (the input word) and one
label (the expected output word).

**Common misunderstanding.**
"Features and labels must be different data types." They do not. Here
both the feature and the label are words. The distinction is purely
about *role*: what goes in vs. what comes out.

---

## 2.7 Model

**Definition.**
A model is the mathematical structure that takes an input, performs
calculations, and produces an output. Before training, the model
contains random parameters and its predictions are meaningless. After
training, its parameters have been adjusted so that its predictions are
(hopefully) useful.

**Why it is needed.**
The model is the thing that actually does the predicting. Everything
else (data, training, evaluation) exists to create and improve the
model.

**Intuition.**
Think of a model as a formula in an Excel spreadsheet with some unknown
coefficients. Training fills in those coefficients using data. Once
filled in, the spreadsheet can take new inputs and produce predictions.

**Example.**
A very simple model: `output = input × w + b`, where `w` and `b` are
parameters the model needs to learn. Before training, `w` and `b` are
random, so the output is meaningless.

**Connection to the assignment.**
The entire forward-pass code *is* the model. It defines a sequence of
mathematical operations: one-hot encoding → hidden layer → activation
→ output layer → softmax. The weights and biases are the parameters.
They start random — which is exactly why the first prediction is random.

**Common misunderstanding.**
"Model = code." Not exactly. The *code* defines the structure. The
*model* is the structure + the learned parameter values. Same code with
different parameters is a different model.

---

## 2.8 Prediction

**Definition.**
A prediction is the output the model produces for a given input. During
training it is compared against the correct label. During deployment it
is the system's best guess.

**Why it is needed.**
Prediction is the purpose of the whole system. Everything we build (data
pipelines, networks, training loops) exists so that the model can make
accurate predictions.

**Intuition.**
If someone asks you "What comes after 'the'?" and you guess "cat", that
guess is your prediction. If the answer key says "cat", your prediction
was correct.

**Example.**
The model receives the word `"the"` and predicts `"sat"`. The correct
answer is `"cat"`. The prediction is wrong this time.

**Connection to the assignment.**
The forward pass ends with a prediction: the word with the highest
probability after softmax. Since the model has not been trained, this
prediction is effectively a random guess.

**Common misunderstanding.**
"A confident prediction is a correct prediction." No. A model can output
a very high probability for the wrong word. Confidence ≠ correctness.

---

# 3. Representing Words as Numbers

## 3.1 Vocabulary

**Definition.**
The vocabulary is the complete, fixed list of words (or tokens) the
model knows about. Every word the model can accept as input or produce
as output must be in this list. Words outside the vocabulary are unknown
to the model.

**Why it is needed.**
Computers work with numbers, not text. Before we can convert a word to
a number, we need a definitive list of all possible words so we can
assign each one a unique number.

**Intuition.**
Think of an enum in C#:

```csharp
enum Vocab { The = 0, Cat = 1, Sat = 2, On = 3, Mat = 4 }
```

This enum is the vocabulary. Any word not in the enum cannot be used.

**Example.**

| Index | Word  |
|------:|-------|
| 0     | "the" |
| 1     | "cat" |
| 2     | "sat" |
| 3     | "on"  |
| 4     | "mat" |

Vocabulary size = 5.

**Connection to the assignment.**
The first thing the forward-pass code does is define a vocabulary. The
vocabulary size determines the dimensions of the input vector, the
output vector, and several weight matrices.

**Common misunderstanding.**
"The vocabulary can grow at inference time." No. If the model was built
for a 5-word vocabulary, it cannot handle a 6th word without being
rebuilt.

---

## 3.2 Token

**Definition.**
A token is the smallest unit of text the model operates on. In simple
models, one token = one word. In production LLMs, tokens can be
sub-words (e.g., "running" → "run" + "ning").

**Why it is needed.**
We need a consistent unit of processing. "Token" is the general term
for that unit, whether it is a word, a sub-word, or even a single
character.

**Intuition.**
Tokens are to an NLP model what rows are to a database: the fundamental
unit of data.

**Example.**
In the sentence "the cat sat", the tokens are: `"the"`, `"cat"`,
`"sat"` — three tokens.

**Connection to the assignment.**
Each input to the network is one token (one word). The model predicts
one token (one word) as output.

**Common misunderstanding.**
"Token always means word." In this assignment it does, but in GPT-style
models a token can be part of a word (e.g., "un" + "believ" + "able").

---

## 3.3 Token Index

**Definition.**
A token index is the integer position of a token in the vocabulary. It
is how we convert a human-readable word into a number the computer can
use.

**Why it is needed.**
Neural networks cannot process the string `"cat"`. They need a number.
The token index provides that number.

**Intuition.**
It is the same as looking up the position of an item in an array:

```csharp
string[] vocab = { "the", "cat", "sat", "on", "mat" };
int index = Array.IndexOf(vocab, "cat"); // returns 1
```

**Example.**
If the vocabulary is `["the", "cat", "sat", "on", "mat"]`:

| Word  | Token index |
|-------|-------------|
| "the" | 0           |
| "cat" | 1           |
| "sat" | 2           |
| "on"  | 3           |
| "mat" | 4           |

**Connection to the assignment.**
The code uses a dictionary (Python `dict`) to map each word to its
index. This index is then used to construct the one-hot vector.

**Common misunderstanding.**
"The index encodes meaning." It does not. Index 0 is not "less than"
index 4. The numbers are arbitrary identifiers, like primary keys in a
database.

---

## 3.4 Scalar

**Definition.**
A scalar is a single number. It has no rows, no columns — just one
value.

**Why it is needed.**
Understanding scalars helps you distinguish them from vectors (lists of
numbers) and matrices (grids of numbers). Shape-tracking through a
neural network requires knowing what is a scalar and what is not.

**Intuition.**
A scalar is like a single cell in an Excel spreadsheet.

**Example.**
`42`, `3.14`, `-0.7` — all scalars.

**Connection to the assignment.**
The token index is a scalar. The predicted index after argmax is a
scalar. The learning rate (in later assignments) is a scalar.

**Common misunderstanding.**
"A one-element array is a scalar." Technically, a scalar has no
dimensions while a one-element array has shape `(1,)`. In practice this
distinction rarely matters, but be aware of it when debugging shape
errors.

---

## 3.5 Vector

**Definition.**
A vector is an ordered list of numbers. In machine learning, it is the primary data structure for representing features, activations, and predictions.

**Why it is needed.**
Vectors are the standard way to represent data in neural networks. The input (one-hot word representation), the hidden-layer activations, the logits, and the softmax probabilities are all vectors.

**Intuition.**
Think of a single row in a database table or a 1-D array in C#:
```csharp
float[] v = { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f };
```

---

### The Dual Meaning of "Dimension"
The word "dimension" is used in two different ways in ML engineering: mathematical dimension and programming (NumPy) dimension. Confusing the two is a major source of shape-mismatch bugs.

#### 1. Mathematical Vector Dimensionality
In mathematics, the dimension of a vector is simply the **number of components** (elements) it contains.
- `[3, 4]` is a 2-dimensional vector (2D space).
- `[1, 2, 3]` is a 3-dimensional vector (3D space).
- `[0, 1, 0, 0, 0]` is a **5-dimensional vector** because it has five components.

#### 2. NumPy Array Dimensionality (`ndim`)
In programming libraries like NumPy, `ndim` means the **number of array axes (grid dimensions)**, not the number of components.
- A flat 1-D array is a single axis. Its NumPy dimension (`ndim`) is always `1`, regardless of how many elements it holds.

```python
import numpy as np

# A 5-dimensional mathematical vector represented as a 1-D NumPy array
v = np.array([0, 1, 0, 0, 0])

print(v.ndim)
# Outputs: 1  (Because it is a 1-D array of numbers)

print(v.shape)
# Outputs: (5,)  (A tuple with 5 elements along its single axis)
```

---

### Shape and Orientation (Row vs. Column)
A raw 1-D vector has no orientation (it is neither a row nor a column; it is just a flat sequence of numbers). However, when we perform matrix multiplication, the math requires us to treat vectors as either:
- **Row Vector:** A matrix with 1 row and N columns. Shape: `(1, N)`. `ndim` = `2`.
- **Column Vector:** A matrix with N rows and 1 column. Shape: `(N, 1)`. `ndim` = `2`.

In NumPy, you can add an axis to convert a flat `(N,)` vector into a 2-D row or column vector:
```python
v_row = v[np.newaxis, :]  # Shape: (1, 5), ndim: 2
v_col = v[:, np.newaxis]  # Shape: (5, 1), ndim: 2
```

<img src="./images/vector_dimensionality.png" alt="Mathematical Vector Dimension vs NumPy Array Dimension" width="600" />

**Connection to the assignment.**
The one-hot encoded input is represented as a flat 1-D vector of shape `(vocab_size,)`. In your assignment, you will compute the hidden layer using matrix multiplication:
```python
hidden_pre_activation = input_vector @ input_to_hidden_weights + hidden_bias
```
Here is the step-by-step logic of how NumPy handles this under the hood:
1. **The Shape Mismatch Problem:** Mathematically, you cannot multiply a 1-D array of shape `(vocab_size,)` directly by a 2-D matrix of shape `(vocab_size, hidden_size)`. Standard matrix multiplication requires the left operand to have 2 dimensions (rows and columns).
2. **NumPy's Automatic Promotion:** To resolve this, when NumPy encounters a 1-D array on the **left** of the `@` operator:
   - It temporarily *promotes* (pads) the vector by prepending a `1` to its shape: `(vocab_size,)` becomes a row vector of shape `(1, vocab_size)`.
3. **The Matrix Math:** Now that shapes are aligned, the multiplication runs:
   $$\text{Shape: } (1, \text{vocab\_size}) \times (\text{vocab\_size}, \text{hidden\_size}) \rightarrow (1, \text{hidden\_size})$$
4. **Automatic Demotion:** Once the multiplication is finished, NumPy automatically *demotes* the result by stripping the prepended `1`, converting the shape from `(1, hidden_size)` back to a clean 1-D vector of shape `(hidden_size,)`.

This convenient behavior means you do not have to write boilerplate code like `input_vector.reshape(1, -1)` before every layer just to satisfy matrix algebra. NumPy treats it as a row vector for the math but returns a clean flat vector for your code.

**Common misunderstanding.**
"A mathematical 5D vector must have shape `(5, 5)` or have `ndim = 5`." No. A 5D vector is a flat list of 5 numbers, which is represented in NumPy as shape `(5,)` and `ndim = 1`. 

---

## 3.6 One-Hot Encoding

**Definition.**
One-hot encoding converts a token index into a vector where exactly one
element is `1` and all others are `0`. The position of the `1`
corresponds to the token index.

**Why it is needed.**
A neural network cannot take an integer like `1` and understand that it
means "cat". One-hot encoding converts the integer into a vector that
the network's matrix multiplication can process.

**Intuition.**
Imagine a row of light switches — one for each word in the vocabulary.
To say "cat", you flip only the switch at position 1 and leave all
others off.

**Example.**
Vocabulary: `["the", "cat", "sat", "on", "mat"]` (size 5).

Token index for `"cat"` = 1.

One-hot vector: `[0, 1, 0, 0, 0]` — shape `(5,)`.

Notice that:
- The vector length equals the vocabulary size.
- There is exactly one `1`.
- The position of the `1` is the token index.

**Connection to the assignment.**
The forward-pass code constructs a one-hot vector from the input word's
token index. This vector becomes the input to the first layer of the
network.

**Common misunderstanding.**
"One-hot vectors capture word meaning." They do not. The vectors for
"cat" and "kitten" are just as different as the vectors for "cat" and
"airplane". One-hot encoding preserves identity but not semantics.
Embeddings (a later assignment) solve this problem.

---

## 3.7 How a Word Becomes a Numerical Representation

This subsection ties together the four concepts above into one clear
pipeline. This is the process the forward-pass code performs before any
neural-network computation begins.

```
Step 1: Start with a word
        "cat"

Step 2: Look up the word in the vocabulary
        vocabulary = ["the", "cat", "sat", "on", "mat"]

Step 3: Get the token index
        "cat" → index 1

Step 4: Create a one-hot vector (length = vocabulary size)
        index 1 → [0, 1, 0, 0, 0]

Step 5: This vector is the neural network's input
        input_vector = [0, 1, 0, 0, 0]   shape: (5,)
```

<img src="./images/word_representation.png" alt="Word Representation Pipeline Diagram" width="600" />

**Why this pipeline exists.**
Neural networks perform matrix multiplication. Matrix multiplication
requires numerical inputs. A string like `"cat"` has no numerical
meaning. This pipeline converts the string into a fixed-length numerical
vector that matrix multiplication can operate on.

**What information is preserved.**
The identity of the word (which word was chosen).

**What information is lost.**
Any notion of similarity between words. "cat" and "kitten" produce
completely different vectors.

---

# 4. Mathematical Foundations

## 4.1 Matrix

**Definition.**
A matrix is a rectangular grid of numbers arranged in rows and columns.
A vector is a special case of a matrix (either one row or one column).

**Why it is needed.**
Neural networks store their learnable parameters (weights) in matrices.
The core operation of a neural network — transforming an input into an
output — is matrix multiplication.

**Intuition.**
Think of a 2-D array in C#, or a table in Excel with rows and columns:

```
[ 0.2  0.5 ]
[ 0.8 -0.1 ]
[ 0.3  0.6 ]
```

This is a 3×2 matrix: 3 rows, 2 columns.

**Example.**

```
A = [ 1  2 ]    shape: (2, 3) — 2 rows, 3 columns
    [ 3  4 ]
    

A = [ 1  2  3 ]    shape: (2, 3) — 2 rows, 3 columns
    [ 4  5  6 ]
```

**Connection to the assignment.**
The weights connecting the input layer to the hidden layer are stored in
a matrix of shape `(vocab_size, hidden_size)`. The weights connecting
the hidden layer to the output layer are stored in a matrix of shape
`(hidden_size, vocab_size)`.

**Common misunderstanding.**
"Matrix and array mean the same thing." In everyday code they are used
interchangeably, but mathematically a matrix has specific rules about
multiplication and transposition. When debugging neural networks, you
think of them as matrices with shapes.

---

## 4.2 Matrix Shape

**Definition.**
The shape of a matrix is a tuple `(rows, columns)` that tells you its
dimensions. The shape of a vector is `(length,)`.

**Why it is needed.**
Shape tracking is the single most important debugging skill in neural
network engineering. If two matrices have incompatible shapes, the
multiplication will fail. Knowing the shape at every step lets you catch
errors before they happen.

**Intuition.**
Shape is like a type signature in C#. If a function expects `int[]` and
you pass `string`, it fails at compile time. Similarly, if a matrix
multiplication expects shape `(5,)` and you provide shape `(3,)`, it
fails at runtime.

**Example.**

| Value                     | Shape     |
|---------------------------|-----------|
| A single number: `42`    | scalar    |
| `[0, 1, 0, 0, 0]`        | `(5,)`   |
| A 5×3 grid of numbers    | `(5, 3)` |
| A 3×5 grid of numbers    | `(3, 5)` |

<img src="./images/matrix_shapes.png" alt="Scalar Vector and Matrix Shapes Diagram" width="600" />

**Connection to the assignment.**
Every variable in the forward pass has a specific shape. Tracking shapes
from input to output is how you verify the code is correct:

```
input_vector:            (vocab_size,)
input_to_hidden_weights: (vocab_size, hidden_size)
hidden_bias:             (hidden_size,)
hidden_activation:       (hidden_size,)
hidden_to_output_weights:(hidden_size, vocab_size)
output_bias:             (vocab_size,)
logits:                  (vocab_size,)
probabilities:           (vocab_size,)
```

**Common misunderstanding.**
"Shape `(5,)` and shape `(1, 5)` are the same." They are not. `(5,)` is
a 1-D vector. `(1, 5)` is a 2-D matrix with one row and five columns.
Most code handles this gracefully, but it can cause subtle bugs.

---

## 4.3 Dot Product

**Definition.**
The dot product takes two vectors of the same length, multiplies
corresponding elements, and sums the results to produce a single
scalar.

**Why it is needed.**
The dot product is the fundamental operation inside matrix
multiplication. Every element in the output of a matrix multiplication
is computed by a dot product.

**Intuition.**
Think of it as a "weighted score". If you have a feature vector
`[hours_studied, practice_tests]` and a weight vector
`[importance_of_study, importance_of_tests]`, the dot product gives you
a single overall score.

**Example.**

```
a = [1, 2, 3]
b = [4, 5, 6]

dot product = (1×4) + (2×5) + (3×6)
            =   4   +  10   +  18
            =  32
```

Two vectors of length 3 produce one scalar.

**Connection to the assignment.**
When the input vector is multiplied by the weight matrix, each column of
the weight matrix is dot-producted with the input vector. This produces
one hidden neuron's value.

**Common misunderstanding.**
"The dot product is the same as element-wise multiplication." Element-
wise multiplication produces a vector; the dot product produces a
scalar. The dot product includes the summation step.

---

## 4.4 Matrix Multiplication

**Definition.**
Matrix multiplication (matmul) takes two matrices and produces a new
matrix. Each element of the result is the dot product of one row from
the first matrix and one column from the second matrix.

**The shape rule:**
If matrix A has shape `(m, n)` and matrix B has shape `(n, p)`, the
result has shape `(m, p)`. The inner dimensions (`n` and `n`) must
match.

**Why it is needed.**
Matrix multiplication is how a neural network transforms its input at
each layer. Input × Weights = Output.

**Intuition.**
Think of it as running the dot product in a loop. If you have one input
vector (shape `(5,)`) and a weight matrix with 3 columns (shape
`(5, 3)`), you perform 3 dot products — one per column — and get a
result vector of shape `(3,)`.

**Example.**

```
input  = [0, 1, 0, 0, 0]    shape: (5,)

weights = [ 0.1  0.4 ]
          [ 0.2  0.5 ]      shape: (5, 2)
          [ 0.3  0.6 ]
          [ 0.7  0.8 ]
          [ 0.9  1.0 ]

result = input × weights

Column 0: (0×0.1) + (1×0.2) + (0×0.3) + (0×0.7) + (0×0.9) = 0.2
Column 1: (0×0.4) + (1×0.5) + (0×0.6) + (0×0.8) + (0×1.0) = 0.5

result = [0.2, 0.5]    shape: (2,)
```

<img src="./images/matrix_multiplication.png" alt="Matrix Multiplication Row Selection Diagram" width="600" />

Notice what happened: the one-hot vector effectively *selected row 1*
from the weight matrix. This is an important insight — multiplying a
one-hot vector by a matrix picks out the row corresponding to the hot
index.

**Connection to the assignment.**
The forward pass performs two matrix multiplications:
1. `input_vector × input_to_hidden_weights` → produces the hidden
   layer's pre-activation values.
2. `hidden_activation × hidden_to_output_weights` → produces the raw
   scores (logits).

**Common misunderstanding.**
"Matrix multiplication is commutative (A × B = B × A)." It is not.
Order matters. A × B and B × A produce different results (and may not
even be compatible).

---

# 5. Neural-Network Building Blocks

## 5.1 Artificial Neuron

**Definition.**
An artificial neuron is a small computational unit that takes one or
more inputs, multiplies each input by a weight, sums the results, adds a
bias, and passes the total through an activation function to produce one
output.

**Formula:**

```
output = activation(w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + bias)
```

<img src="./images/artificial_neuron.png" alt="Artificial Neuron Diagram" width="600" />

**Why it is needed.**
The neuron is the building block of every neural network. A single
neuron can learn a simple linear boundary. Groups of neurons (layers)
can learn complex, non-linear patterns.

**Intuition.**
Think of a neuron as a tiny decision-maker. It receives several signals
(inputs), weighs their importance (weights), makes a combined decision
(weighted sum + bias), and outputs a signal (activation).

**Example.**

```
Inputs:  x₁ = 0.5, x₂ = 0.3
Weights: w₁ = 0.8, w₂ = 0.2
Bias:    b  = 0.1

Weighted sum = (0.5 × 0.8) + (0.3 × 0.2) + 0.1
             = 0.4 + 0.06 + 0.1
             = 0.56

Output (before activation) = 0.56
```

**Connection to the assignment.**
The hidden layer in the forward pass contains multiple neurons. Each
hidden neuron receives the entire input vector, multiplies it by its
weights, adds its bias, and outputs one value.

**Common misunderstanding.**
"An artificial neuron mimics a biological neuron exactly." It does not.
It is a loose mathematical inspiration, not a biological simulation.

---

## 5.2 Weights

**Definition.**
Weights are the learnable parameters that determine how much influence
each input has on a neuron's output. They are stored in weight matrices.

**Why they are needed.**
Weights are what the model *learns*. Before training, they are random.
After training, they encode the patterns discovered in the data. Without
weights, every input would produce the same output.

**Intuition.**
Think of weights as volume knobs on a mixing console. Each knob
controls how loud one channel (input) is in the final mix (output).
Training adjusts these knobs until the mix sounds right (the model's
predictions are accurate).

**Example.**
If `weight = 0.9`, the input has a strong influence.
If `weight = 0.01`, the input is nearly ignored.
If `weight = -0.5`, the input has a moderate *negative* influence
(pushing the output down).

**Connection to the assignment.**
The forward pass has two weight matrices:
- `input_to_hidden_weights`: shape `(vocab_size, hidden_size)`
- `hidden_to_output_weights`: shape `(hidden_size, vocab_size)`

These are initialised randomly and are the values that would change
during training.

**Common misunderstanding.**
"Higher weight = better." A negative weight is not bad — it simply means
that input pushes the output in the opposite direction. The sign carries
information.

---

## 5.3 Bias

**Definition.**
A bias is an additional learnable parameter added to the weighted sum
before the activation function. Each neuron has its own bias value.

**Formula with bias:**

```
z = (inputs × weights) + bias
```

**Why it is needed.**
Without bias, the neuron's output must pass through the origin
(zero input always produces zero output). The bias shifts the output,
giving the neuron the flexibility to activate even when all inputs are
zero.

**Intuition (C# analogy).**
Think of a linear equation: `y = mx + b`. The `m` is the weight and the
`b` is the bias. Without `b`, the line must pass through the origin. The
bias lets the line shift up or down.

**Example.**

```
weighted_sum = 0.0   (all inputs happen to cancel out)
bias         = 0.3

with bias:    z = 0.0 + 0.3 = 0.3   → neuron can still activate
without bias: z = 0.0               → neuron is stuck at zero
```

**Connection to the assignment.**
The forward pass has two bias vectors:
- `hidden_bias`: shape `(hidden_size,)` — added after
  input-to-hidden multiplication.
- `output_bias`: shape `(vocab_size,)` — added after
  hidden-to-output multiplication.

**Common misunderstanding.**
"Bias is optional." Technically the network can train without bias, but
removing it reduces the model's capacity and can prevent it from
learning certain patterns.

---

## 5.4 Random Initialization

**Definition.**
Random initialization means setting all weights and biases to small
random numbers before training begins.

**Why it is needed.**
If all weights started at zero, every neuron would compute the same
value. They would all learn the same thing and the network would
effectively have only one neuron. Random initialization breaks this
symmetry so each neuron can specialise.

**Intuition.**
Imagine sending ten detectives to investigate a crime. If they all start
at the same location and follow the same clues, they cover the same
ground. If each starts at a different random location, they explore
different parts of the city and collectively find more evidence.

**Example.**

```python
weights = np.random.randn(5, 3) * 0.1
```

This produces a 5×3 matrix where each value is a small random number
drawn from a normal distribution.

**Connection to the assignment.**
The forward-pass code initialises all weights randomly. This is why the
first prediction is essentially a random guess — the weights have not
been trained.

**Common misunderstanding.**
"Random initialization means the model is broken." It is not broken — it
is just untrained. The random values are a *starting point*, not a bug.

---

## 5.5 Hidden Layer

**Definition.**
A hidden layer is a group of neurons between the input layer and the
output layer. It is called "hidden" because its values are not directly
visible in the input or output — they are intermediate computations.

**Why it is needed.**
A single layer (input → output) can only learn linear relationships.
Hidden layers allow the network to learn non-linear patterns — patterns
that are curves, not straight lines.

**Intuition (API analogy).**
Think of a three-tier architecture:

```
Client (Input) → Business Logic (Hidden) → Database (Output)
```

The business logic layer transforms the raw request into something the
database can use. Similarly, the hidden layer transforms the raw input
into a more useful representation before passing it to the output layer.

**Example.**
If the input vector has 5 elements and the hidden layer has 3 neurons,
then:
- Input shape: `(5,)`
- Weight matrix shape: `(5, 3)`
- Hidden layer output shape: `(3,)`

The 5-element input is transformed into a 3-element intermediate
representation.

**Connection to the assignment.**
The forward-pass code has exactly one hidden layer. The computation is:

```
hidden = activation(input_vector × input_to_hidden_weights + hidden_bias)
```

**Common misunderstanding.**
"More hidden layers are always better." Adding layers makes the network
harder to train. The right number of layers depends on the problem
complexity. For this assignment, one hidden layer is sufficient.

---

## 5.6 Activation Function

**Definition.**
An activation function is a mathematical function applied to a neuron's
output after the weighted sum + bias. It introduces non-linearity into
the network.

**Why it is needed.**
Without an activation function, stacking multiple layers is pointless —
the result is still a linear transformation (you could collapse all
layers into a single multiplication). The activation function makes each
layer meaningfully different from the next.

**Intuition.**
Think of an activation function as a filter or a gate. The neuron
computes a raw score, and the activation function decides how to reshape
that score before passing it on.

Without activation functions, a 100-layer network would be
mathematically equivalent to a single-layer network. It is the non-
linearity that gives depth its power.

**Example.**
Common activation functions:

| Function | Formula           | Output range    |
|----------|-------------------|-----------------|
| Sigmoid  | 1 / (1 + e^(-x))  | (0, 1)          |
| tanh     | (e^x - e^(-x)) / (e^x + e^(-x)) | (-1, 1) |
| ReLU     | max(0, x)         | [0, ∞)          |

**Connection to the assignment.**
The forward pass uses `tanh` as its activation function on the hidden
layer.

**Common misunderstanding.**
"The activation function is applied to the output layer too." In this
assignment, the output layer uses *softmax*, which serves a different
purpose (converting raw scores to probabilities). The term "activation
function" usually refers to the function applied in hidden layers.

---

## 5.7 tanh

**Definition.**
`tanh` (hyperbolic tangent) is an activation function that squashes any
input value into the range `(-1, 1)`.

**Formula:**

```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
```

**Why it is needed.**
It introduces non-linearity (essential, as discussed above) and also
centres the output around zero. This zero-centred property can help
during training because it keeps values balanced.

**Intuition.**
Imagine a thermostat that converts any temperature reading into a value
between -1 and 1:
- Very cold → close to -1
- Room temperature → close to 0
- Very hot → close to 1

No matter how extreme the input, the output stays bounded.

**Example.**

```
tanh(-100) ≈ -1.0
tanh(-1)   ≈ -0.76
tanh(0)    =  0.0
tanh(1)    ≈  0.76
tanh(100)  ≈  1.0
```

**Connection to the assignment.**
After `input_vector × input_to_hidden_weights + hidden_bias`, the
forward pass applies `tanh` to every element:

```
hidden_activation = tanh(hidden_pre_activation)
```

This squashes the hidden layer values into `(-1, 1)`.

**Common misunderstanding.**
"tanh output can be exactly -1 or 1." It cannot. It approaches -1 and 1
asymptotically (gets infinitely close but never reaches them). In
practice, for very large or very small inputs, it is close enough that
floating-point arithmetic rounds to -1 or 1.

---

# 6. Producing a Prediction

## 6.1 Raw Scores (Logits)

**Definition.**
After the hidden layer's output is multiplied by the second weight
matrix and the output bias is added, the result is a vector of numbers
— one number per word in the vocabulary. These numbers are called
**raw scores** or, in technical terminology, **logits**.

- **"Raw scores"** is the intuitive description: unnormalized numbers
  that indicate how much the network favours each word.
- **"Logits"** is the standard technical term used in papers and
  frameworks.

They mean the same thing. This guide uses both terms interchangeably.

**Why they are needed.**
The raw scores are the network's "opinion" before any normalisation.
They can be any value — positive, negative, or zero. They are not yet
probabilities.

**Intuition.**
Imagine a panel of judges scoring contestants. Each judge scribbles a
raw number on a card: `+3.2`, `-1.5`, `+0.8`. These are not yet
percentages — they are just gut-feeling scores. They need to be
converted to percentages before we can compare them fairly. That
conversion is what softmax does (next section).

**Example.**

```
logits = [2.1, 0.3, -1.0, 0.5, -0.2]
```

The network slightly favours word 0 (score 2.1) and dislikes word 2
(score -1.0), but these are not probabilities.

**Connection to the assignment.**
The forward pass computes logits as:

```
logits = hidden_activation × hidden_to_output_weights + output_bias
```

Shape: `(vocab_size,)`.

**Common misunderstanding.**
"A logit of 2.1 means 210% confidence." Logits have no bounded meaning
on their own. Their magnitude only becomes meaningful *relative to each
other* after softmax.

---

## 6.2 Softmax

**Definition.**
Softmax converts a vector of raw scores (logits) into a probability
distribution — a vector of positive numbers that sum to exactly 1.

**Formula:**

```
softmax(xᵢ) = eˣⁱ / Σⱼ eˣʲ
```

For each element: raise *e* (≈ 2.718) to the power of the element, then
divide by the sum of *e* raised to the power of *all* elements.

**Why it is needed.**
Logits are unbounded and hard to interpret. Softmax converts them into
probabilities so we can say "the network is 65% confident the next word
is 'cat'."

**Intuition.**
Think of softmax as a "normalise and make positive" function. It takes
any list of numbers (positive, negative, mixed) and converts them into a
valid probability distribution.

**Example.**

```
logits = [2.0, 1.0, 0.1]

e^2.0 = 7.389
e^1.0 = 2.718
e^0.1 = 1.105

sum   = 7.389 + 2.718 + 1.105 = 11.212

softmax = [7.389/11.212, 2.718/11.212, 1.105/11.212]
        = [0.659,        0.242,        0.099        ]

Sum = 0.659 + 0.242 + 0.099 = 1.000 ✓
```

The largest logit (2.0) gets the highest probability (0.659).

**Connection to the assignment.**
The forward pass applies softmax to the logits to produce probabilities:

```
probabilities = softmax(logits)    shape: (vocab_size,)
```

**Common misunderstanding.**
"The highest probability is always correct." No. On an untrained model,
the probabilities are determined by random weights. A high probability
means the random numbers happened to favour that word, not that the
model "knows" the answer.

---

## 6.3 Probability Distribution

**Definition.**
A probability distribution is a set of non-negative numbers that sum
to 1. Each number represents the probability (likelihood) of one
outcome.

**Why it is needed.**
It gives us a standardised way to compare the network's confidence
across all possible outputs. Without it, raw scores are not directly
comparable.

**Intuition.**
Think of a pie chart. Each slice represents one word's probability. All
slices together make up 100% of the pie.

**Example.**

```
probabilities = [0.05, 0.65, 0.10, 0.15, 0.05]
                  the   cat   sat    on   mat
```

The model is 65% confident the next word is "cat". All values are ≥ 0
and they sum to 1.0.

**Connection to the assignment.**
The softmax output in the forward pass is a probability distribution
over the vocabulary: one probability for each word.

**Common misunderstanding.**
"A uniform distribution means the model is broken." A uniform
distribution (e.g., `[0.2, 0.2, 0.2, 0.2, 0.2]` for 5 words) just
means the model has no preference. This is expected for an untrained
model with balanced random weights.

---

## 6.4 Argmax

**Definition.**
Argmax returns the *index* of the highest value in a vector. (Not the
value itself — the index.)

**Why it is needed.**
After softmax produces probabilities, we need to pick the single most
likely word. Argmax does this by finding which index has the highest
probability.

**Intuition.**
It is `Array.IndexOf(array, array.Max())` in C#:

```csharp
float[] probs = { 0.05f, 0.65f, 0.10f, 0.15f, 0.05f };
int predicted = Array.IndexOf(probs, probs.Max()); // returns 1
```

**Example.**

```
probabilities = [0.05, 0.65, 0.10, 0.15, 0.05]
argmax(probabilities) = 1
```

Index 1 → look up vocabulary → "cat".

**Connection to the assignment.**
The last step of the forward pass is:

```
predicted_index = argmax(probabilities)
predicted_word  = vocabulary[predicted_index]
```

This is how the network produces its final prediction.

**Common misunderstanding.**
"Argmax returns the highest value." It returns the *index* (position) of
the highest value. If you want the value itself, that is just `max`.

---

# 7. The Complete Forward Pass

## 7.1 Input-to-Output Flow

The forward pass is the complete sequence of computations from input
word to predicted word. There is no learning, no error correction — just
a one-way flow of data through the network.

```
Word
  → Vocabulary look-up → Token Index (scalar)
  → One-Hot Encoding   → Input Vector (vocab_size,)
  → × Input-to-Hidden Weights + Hidden Bias
                        → Hidden Pre-Activation (hidden_size,)
  → tanh               → Hidden Activation (hidden_size,)
  → × Hidden-to-Output Weights + Output Bias
                        → Logits (vocab_size,)
  → Softmax            → Probabilities (vocab_size,)
  → Argmax             → Predicted Index (scalar)
  → Vocabulary look-up → Predicted Word
```

<img src="./images/forward_pass_flow.png" alt="Complete Forward Pass Flow Diagram" width="600" />

Every arrow represents one computation. Every computation has a defined
input shape and output shape.

---

## 7.2 Shape Tracing Through the Network

For a vocabulary of 5 words and a hidden layer of 3 neurons:

| Step | Value                      | Shape   |
|------|----------------------------|---------|
| 1    | Token index                | scalar  |
| 2    | Input vector (one-hot)     | (5,)    |
| 3    | Input-to-hidden weights    | (5, 3)  |
| 4    | Hidden bias                | (3,)    |
| 5    | Hidden pre-activation      | (3,)    |
| 6    | Hidden activation (tanh)   | (3,)    |
| 7    | Hidden-to-output weights   | (3, 5)  |
| 8    | Output bias                | (5,)    |
| 9    | Logits                     | (5,)    |
| 10   | Probabilities (softmax)    | (5,)    |
| 11   | Predicted index (argmax)   | scalar  |

Notice the pattern: the shapes go `5 → 3 → 5`. The network compresses
the input into a smaller hidden representation (5→3) and then expands
it back to the output size (3→5).

---

## 7.3 Inference

**Definition.**
Inference is the act of running data through a trained (or untrained)
model to get a prediction. No weights are changed during inference —
it is purely a read-only forward computation.

**Intuition.**
Training is like studying for an exam. Inference is like taking the
exam. During the exam, you do not learn new material — you use what
you already know.

**Connection to the assignment.**
The entire forward-pass assignment is inference. You run an input
through the network and observe the output. No learning happens.

---

## 7.4 Untrained-Model Behaviour

**Definition.**
An untrained model is one whose weights are still at their initial
random values. It has not seen any data, so its predictions are not
based on learned patterns.

**Why this matters.**
Understanding that an untrained model produces random predictions is
critical. It sets the baseline. Everything that follows in future
assignments (loss functions, backpropagation, training loops) exists
to move the model from random predictions to accurate ones.

**Connection to the assignment.**
The forward-pass assignment deliberately shows you an untrained model.
The point is to understand: this is what the network does *before* it
learns. The output is essentially a random guess.

---

## 7.5 Why the First Prediction Is Random

The prediction depends entirely on the weights and biases. Since these
are initialised randomly, the logits are random, the softmax
probabilities are random, and the argmax picks whichever random
probability happens to be highest.

**Can an untrained model be accidentally correct?**
Yes. With a 5-word vocabulary, there is a 20% chance the random
prediction matches the correct answer — like rolling a 5-sided die and
guessing correctly.

**What changes during training?**
Only the weights and biases. The forward-pass code (the structure) stays
the same. Training adjusts the weight values so that the logits for the
correct word become larger than the logits for incorrect words.

---

# 8. Complete Worked Numerical Example

This section traces the word `"cat"` through a tiny network from start
to finish. All numbers are chosen for clarity, not realism.

**Network dimensions:**
- Vocabulary size: 3 (`"the"`, `"cat"`, `"sat"`)
- Hidden size: 2

**Shapes at a glance:**

| Value                      | Shape   |
|----------------------------|---------|
| Input vector               | (3,)    |
| Input-to-hidden weights    | (3, 2)  |
| Hidden bias                | (2,)    |
| Hidden pre-activation      | (2,)    |
| Hidden activation          | (2,)    |
| Hidden-to-output weights   | (2, 3)  |
| Output bias                | (3,)    |
| Logits                     | (3,)    |
| Probabilities              | (3,)    |

---

## Step 1 — Token Index

```
Vocabulary: { "the": 0, "cat": 1, "sat": 2 }
Input word: "cat"
Token index: 1
```

---

## Step 2 — One-Hot Encoding

```
input_vector = [0, 1, 0]    shape: (3,)
```

Position 1 is `1` (for "cat"), all others are `0`.

---

## Step 3 — Input-to-Hidden Multiplication

```
input_to_hidden_weights = [ 0.1   0.4 ]
                          [ 0.2   0.5 ]     shape: (3, 2)
                          [ 0.3   0.6 ]

hidden_bias = [0.01, -0.02]                 shape: (2,)
```

Computation:

```
pre_activation = input_vector × weights + bias

Column 0: (0×0.1) + (1×0.2) + (0×0.3) + 0.01 = 0.2 + 0.01 = 0.21
Column 1: (0×0.4) + (1×0.5) + (0×0.6) - 0.02 = 0.5 - 0.02 = 0.48

hidden_pre_activation = [0.21, 0.48]        shape: (2,)
```

Note: Because the input is one-hot with the `1` at index 1, we
effectively selected row 1 of the weight matrix (`[0.2, 0.5]`) and
added the bias.

---

## Step 4 — Activation (tanh)

```
tanh(0.21) ≈ 0.2070
tanh(0.48) ≈ 0.4462

hidden_activation = [0.2070, 0.4462]        shape: (2,)
```

Both values are squashed into the range (-1, 1).

---

## Step 5 — Hidden-to-Output Multiplication

```
hidden_to_output_weights = [ 0.3  -0.1   0.2 ]
                           [ 0.4   0.6  -0.3 ]    shape: (2, 3)

output_bias = [0.05, 0.02, -0.01]                  shape: (3,)
```

Computation:

```
logits = hidden_activation × weights + output_bias

Word 0 ("the"): (0.2070×0.3) + (0.4462×0.4) + 0.05
              =  0.0621 + 0.1785 + 0.05
              =  0.2906

Word 1 ("cat"): (0.2070×-0.1) + (0.4462×0.6) + 0.02
              = -0.0207 + 0.2677 + 0.02
              =  0.2670

Word 2 ("sat"): (0.2070×0.2) + (0.4462×-0.3) - 0.01
              =  0.0414 + (-0.1339) - 0.01
              = -0.1025

logits = [0.2906, 0.2670, -0.1025]          shape: (3,)
```

---

## Step 6 — Softmax

```
e^0.2906  = 1.3373
e^0.2670  = 1.3060
e^-0.1025 = 0.9026

sum = 1.3373 + 1.3060 + 0.9026 = 3.5459

probabilities = [1.3373/3.5459, 1.3060/3.5459, 0.9026/3.5459]
              = [0.3771,        0.3683,         0.2546       ]

Sum check: 0.3771 + 0.3683 + 0.2546 = 1.0000 ✓
```

Shape: `(3,)`.

---

## Step 7 — Argmax

```
probabilities = [0.3771, 0.3683, 0.2546]
                  "the"   "cat"   "sat"

argmax = 0   (index of the highest value, 0.3771)
```

---

## Step 8 — Predicted Word

```
predicted_index = 0
predicted_word  = vocabulary[0] = "the"
```

---

## Result

| Field           | Value      |
|-----------------|------------|
| Input word      | "cat"      |
| Predicted word  | "the"      |
| Correct word    | (depends on training pair) |
| Was it correct? | Unlikely — the model is untrained |

The prediction `"the"` was determined entirely by random weights. The
probabilities `[0.38, 0.37, 0.25]` are roughly equal, confirming that
the model has no real preference — it is guessing.

---

# 9. Quick-Revision Glossary

| Term                    | One-line definition |
|-------------------------|---------------------|
| Machine learning        | Systems that learn patterns from data instead of explicit rules. |
| Supervised learning     | Learning from input–output pairs with known labels. |
| Dataset                 | Collection of examples used for training or evaluation. |
| Feature                 | An input property the model uses to make predictions. |
| Label                   | The correct answer the model tries to predict. |
| Model                   | Mathematical structure + learned parameters that produces predictions. |
| Prediction              | The output the model produces for a given input. |
| Vocabulary              | The fixed set of all words the model can process. |
| Token                   | The smallest unit of text the model operates on. |
| Token index             | The integer position of a token in the vocabulary. |
| Scalar                  | A single number with no dimensions. |
| Vector                  | An ordered list of numbers (one dimension). |
| One-hot encoding        | A vector with one `1` and all other elements `0`. |
| Matrix                  | A 2-D grid of numbers with rows and columns. |
| Matrix shape            | The `(rows, columns)` tuple describing a matrix's dimensions. |
| Dot product             | Multiply corresponding elements of two vectors, then sum. |
| Matrix multiplication   | Repeated dot products producing a new matrix or vector. |
| Artificial neuron       | A unit that computes: `activation(inputs × weights + bias)`. |
| Weights                 | Learnable parameters controlling input influence. |
| Bias                    | A learnable offset added before the activation function. |
| Random initialization   | Setting weights to small random values before training. |
| Hidden layer            | Neurons between input and output that learn intermediate features. |
| Activation function     | Non-linear function applied after the weighted sum. |
| tanh                    | Activation function that squashes values to (-1, 1). |
| Raw scores / Logits     | Unnormalized output scores before softmax. |
| Softmax                 | Converts logits into a probability distribution summing to 1. |
| Probability distribution| A set of non-negative numbers that sum to 1. |
| Argmax                  | Returns the index of the largest value in a vector. |
| Forward pass            | One-way computation from input to prediction (no learning). |
| Inference               | Running data through a model to get a prediction. |
| Shape tracing           | Tracking the dimensions of every value through the network. |
| Untrained model         | A model with random weights that has not yet learned from data. |

---

# 10. Readiness Questions

If you can answer these comfortably, you are ready for the assignment.

**Machine-learning foundations:**
1. Why would we use machine learning instead of writing explicit rules?
2. What does "supervised" mean in supervised learning?
3. What is the role of a dataset in machine learning?
4. What is the difference between a feature and a label?
5. What is the difference between a model's structure and its parameters?

**Representing words as numbers:**
6. Why can a neural network not accept the string `"cat"` directly?
7. What is the relationship between vocabulary size and one-hot vector length?
8. Does the one-hot vector for `"cat"` contain any information about what a cat is?

**Mathematical foundations:**
9. What shape results from multiplying a `(5,)` vector by a `(5, 3)` matrix?
10. What does multiplying a one-hot vector by a weight matrix effectively do?

**Neural-network building blocks:**
11. What would happen if all weights were initialised to zero instead of randomly?
12. Why does the hidden layer apply tanh?
13. What range of values can tanh produce?

**Producing a prediction:**
14. Are logits probabilities? Why or why not?
15. What does softmax guarantee about its output?
16. Does argmax return a value or an index?

**The complete forward pass:**
17. Can you trace the shape of every value from input to output?
18. Why is the untrained model's first prediction essentially random?
19. What would need to change for the model to make accurate predictions?

---

# 11. Concepts Deliberately Deferred

The following concepts are related to the forward pass but are **not**
required to understand it. They will be introduced when the learning
process (training) is added in a future assignment.

| Concept                     | Why it is deferred |
|-----------------------------|---------------------|
| Loss functions              | Needed for measuring prediction error (training). |
| Cross-entropy loss          | The specific loss function used with softmax (training). |
| Backpropagation             | Algorithm for computing how to adjust weights (training). |
| Gradients                   | The direction and magnitude of weight adjustments (training). |
| Gradient descent            | The optimisation strategy that uses gradients (training). |
| Stochastic gradient descent | A variant of gradient descent using random batches (training). |
| Adam optimizer              | A modern optimizer combining several techniques (training). |
| Training loop               | The repeated cycle of predict → measure error → adjust (training). |
| Batch processing            | Processing multiple examples at once for efficiency (training). |
| Regularization              | Techniques to prevent overfitting (training). |
| Embeddings                  | Dense vector representations that capture word meaning (future assignment). |

You do not need to understand any of these before starting the
forward-pass assignment.