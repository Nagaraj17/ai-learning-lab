# REVIEW (Week 4: Multi-Head Attention)

Use this review after:

1. studying the Week 4 guide and Topics 26-28;
2. completing both manual exercises;
3. running the Week 4 implementation and saving its evidence.

Answer from memory first. For mathematical questions, show intermediate work
rather than writing only the final number.

## Part 1: Conceptual Understanding

1. What precise limitation of one attention head motivates Multi-Head
   Attention?
2. Why is it incorrect to say that different heads process different subsets
   of tokens?
3. What causes two heads to produce different Query, Key, and Value matrices
   from the same input sequence?
4. Why is one wider "super head" not equivalent to several smaller heads?
5. Why is
   $d_k=d_v=d_{\text{model}}/h$
   a common convention rather than the mathematical definition of MHA?
6. What does feature-axis concatenation preserve?
7. What can the learned output projection $\mathbf{W}_O$ do that
   concatenation alone cannot?
8. Explain the difference between:
   - different attention patterns;
   - head specialization;
   - functional head redundancy.
9. Why does a different-looking heatmap not prove that a head is useful?
10. Why are attention weights not automatically a complete explanation of a
    prediction?

## Part 2: Shapes and Matrix Operations

### Question 1: Complete shape trace

Let:

$$
T=5,\qquad d_{\text{model}}=12,\qquad h=3,\qquad d_k=d_v=4
$$

Write the shape of:

1. $\mathbf{X}$;
2. $\mathbf{W}_Q^{(1)}$, $\mathbf{W}_K^{(1)}$, and
   $\mathbf{W}_V^{(1)}$;
3. $\mathbf{Q}^{(1)}$, $\mathbf{K}^{(1)}$, and
   $\mathbf{V}^{(1)}$;
4. $\mathbf{Q}^{(1)}{\mathbf{K}^{(1)}}^\top$;
5. $\mathbf{A}^{(1)}$;
6. $\mathbf{H}^{(1)}$;
7. all explicit head outputs stored as $(h,T,d_v)$;
8. the concatenated matrix $\mathbf{C}$;
9. $\mathbf{W}_O$;
10. final output $\mathbf{Y}$.

For every matrix multiplication, state why its inner dimensions match.

### Question 2: Concatenation axis

Two head outputs both have shape $(5,4)$.

1. What shape results from concatenating along the feature axis?
2. What incorrect shape results from stacking along the token axis?
3. Explain what semantic mistake the incorrect operation makes.

### Question 3: Pairwise attention distance

Two heads produce:

$$
\mathbf{A}^{(1)}
=
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

$$
\mathbf{A}^{(2)}
=
\begin{bmatrix}
0.5 & 0.5 \\
0.5 & 0.5
\end{bmatrix}
$$

Calculate:

$$
D_{1,2}
=
\frac{1}{4}
\sum_{q=1}^{2}
\sum_{k=1}^{2}
\left|
A^{(1)}_{q,k}-A^{(2)}_{q,k}
\right|
$$

Then answer:

1. Which head is more concentrated?
2. What is the normalized row entropy of Head 1?
3. What is the normalized row entropy of Head 2?
4. What does this comparison establish?
5. What does it not establish?

### Question 4: Ablation delta

A trained model produces these held-out losses:

| Condition | Loss |
| :--- | ---: |
| All heads active | $0.420$ |
| Head 1 ablated | $0.430$ |
| Head 2 ablated | $0.580$ |

Calculate:

$$
\Delta L_1=L_{-1}-L_{\text{base}}
$$

and:

$$
\Delta L_2=L_{-2}-L_{\text{base}}
$$

Then answer:

1. Which head has the stronger measured individual contribution?
2. Is Head 1 proven universally useless?
3. What model, data, and experimental details must accompany the conclusion?

## Part 3: Code Comprehension and Implementation

1. List the complete loop-based NumPy forward pass from $\mathbf{X}$ to
   $\mathbf{Y}$ in the correct order.
2. Why should the implementation return both contextual head outputs and
   attention matrices?
3. Starting from head outputs with shape $(h,T,d_v)$, explain why the code must
   transpose to $(T,h,d_v)$ before reshaping to $(T,h d_v)$.
4. What bug occurs if two $(T,d_v)$ heads are concatenated along axis $0$?
5. Where must a causal mask be applied inside MHA?
6. Why should Softmax subtract each row's maximum score before
   exponentiation?
7. To ablate Head 2, which tensor should be zeroed and at what point in the
   forward pass?
8. Why must the remaining trained parameters stay fixed during an immediate
   ablation test?
9. Which assertions would you add for:
   - input width;
   - number of heads;
   - attention row sums;
   - concatenated width;
   - final output width?

## Part 4: Debugging and Evidence Reasoning

1. A student says, "Head 1 sees the first half of the sequence and Head 2 sees
   the second half." Correct the statement precisely.
2. Two $(4,2)$ head outputs produce a concatenated shape of $(8,2)$. What
   operation was probably wrong?
3. Two trained heads have pairwise attention distance $0.01$, but ablating one
   increases loss substantially. Give two reasons why similar attention
   matrices may still have different functional effects.
4. Two heads have very different heatmaps, but ablating either changes loss by
   almost zero. What conclusions are reasonable?
5. An untrained random forward pass produces different-looking heads. What has
   been demonstrated, and what has not?
6. A head has normalized entropy near $0$. Why does that not prove high
   prediction confidence?
7. Why should causal masked cells be excluded carefully when comparing heads
   across sequences with different valid lengths?
8. Why is selecting heads using the final test set a form of leakage?

## Part 5: Experiment Design

Design a Week 4 experiment that answers:

> Did any trained head develop a consistent and useful relationship pattern?

Your design must state:

1. the training and held-out sequences;
2. the prediction target and loss;
3. the random seeds;
4. what is saved before training;
5. what is saved after training;
6. how heatmaps are compared;
7. how token groups are defined before inspection;
8. how head ablation is performed;
9. which metric determines degradation;
10. how expectation is separated from observation.

Explain why one selected sequence and one random seed are not sufficient.

## Part 6: Transformer-Block Preview

1. If both $\mathbf{X}$ and the MHA output $\mathbf{Y}$ have shape
   $(T,d_{\text{model}})$, why is the residual addition
   $\mathbf{X}+\mathbf{Y}$ valid?
2. At a high level, what does the residual path preserve?
3. At a high level, what does Layer Normalization control?
4. Does Layer Normalization combine information between token positions?
5. Why is the exact ordering of attention, residual addition, and
   normalization deferred to the later Transformer-block topic?

## Part 7: Teach-Back

Explain Week 4 to a learner who understands one-head self-attention.

Your explanation must include:

1. the problem with only one attention distribution;
2. why every head receives the full sequence;
3. how independent projections create different views;
4. why several heads are not equivalent to one wider head;
5. how concatenation and $\mathbf{W}_O$ combine the outputs;
6. why specialization is possible but not guaranteed;
7. how heatmaps, numerical comparison, and ablation provide different kinds of
   evidence;
8. one limitation that remains after MHA.

## Completion Standard

The review is complete only when you can:

- answer the conceptual questions without relying on role-based analogies;
- trace every shape without guessing;
- calculate the distance and ablation examples by hand;
- identify the corresponding NumPy operations;
- design an evidence-based experiment;
- state conclusions without exceeding the measurements.
