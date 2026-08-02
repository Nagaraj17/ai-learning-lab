# REVIEW (Week 4: Multi-Head Attention)

Use these questions after studying the Week 4 notes and running the Week 4
code.

## Part 1: Conceptual Understanding

1. What limitation of single-head attention motivates Multi-Head Attention?
2. Why is it incorrect to say that different heads process different subsets of
   tokens?
3. In one sentence, what does `W_O` do?
4. Why is `d_k = d_model / h` best described as a common convention rather than
   a universal mathematical requirement?
5. Why should we avoid assigning business labels such as "Inventory head" or
   "Pricing head" without measured evidence?

## Part 2: Mathematics and Shape Tracing

1. Let `T = 4`, `d_model = 8`, `h = 2`, `d_k = d_v = 4`.
   - What is the shape of `X`?
   - What is the shape of `W_Q^1`?
   - What is the shape of `Q_1 K_1^T`?
   - What is the shape of `head_1`?
   - What is the shape after concatenating both heads?
   - What is the shape of `W_O` if the final output must return to
     `(T, d_model)`?
2. Why is the multiplication `Q_i K_i^T` valid?
3. Why is the multiplication `A_i V_i` valid?

## Part 3: Code and Implementation

1. In a loop-based NumPy implementation, list the major operations in the
   correct order from `X` to the final multi-head output.
2. Why is a loop-based implementation a better teaching starting point than a
   fully vectorized reshape-transpose implementation?
3. What shape change does `np.concatenate([head_1, head_2], axis=1)` perform
   when both heads have shape `(T, d_v)`?

## Part 4: Debugging and What-If Reasoning

1. A student says, "Head 1 sees the first half of the sequence and Head 2 sees
   the second half." What exactly is wrong with that statement?
2. A student concatenates two `(4, 2)` head outputs and gets `(8, 2)`.
   - What probably went wrong?
3. A trained two-head model loses almost no performance when head 2 is zeroed.
   - What are at least two reasonable interpretations?
4. If a random untrained NumPy forward pass produces two different-looking
   attention matrices, why is that still not evidence of meaningful
   specialization?

## Part 5: Teach-Back Prompts

1. Explain Multi-Head Attention to a Week 3 learner without using the word
   "subspace."
2. Explain why `W_O` is needed using only matrix-shape language.
3. Explain the difference between:
   - what Multi-Head Attention allows
   - what we expect after training
   - what a specific experiment actually proved
