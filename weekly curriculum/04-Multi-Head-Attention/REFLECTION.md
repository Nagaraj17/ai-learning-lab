# REFLECTION (Week 4: Multi-Head Attention)

Use this file to record evidence from the Week 4 implementation.

Do not complete a field from expectation or memory. If a value was not
measured, write **Not measured**. If an artifact was not created, write
**Not created**.

The tables provide space for up to four heads. When the implementation uses
fewer heads, mark unused rows **Not applicable** rather than leaving them
ambiguous.

## 1. Experiment Identity

| Field | Recorded value |
| :--- | :--- |
| Date | |
| Implementation file or notebook | |
| Git commit | |
| Dataset or sequence file | |
| Training split description | |
| Held-out split description | |
| Prediction target | |
| Loss function | |
| Number of random seeds | |

### Model configuration

| Parameter | Value |
| :--- | :--- |
| Sequence length $T$ | |
| Vocabulary size | |
| Model width $d_{\text{model}}$ | |
| Number of heads $h$ | |
| Query/key width $d_k$ | |
| Value width $d_v$ | |
| Learning rate | |
| Epochs | |
| Causal mask used? | |
| Residual connection implemented? | No, unless explicitly added as an optional preview |
| Layer Normalization implemented? | No, unless explicitly added as an optional preview |

The Week 4 core implementation does not require residual connections or Layer
Normalization. Their purpose and shape connection are learned at a high level;
their full implementation belongs to the later Transformer-block work.

## 2. Saved Artifact Manifest

Record paths relative to the repository when possible.

| Artifact | Path | Created? |
| :--- | :--- | :---: |
| NumPy mechanics output | | |
| Training-loss history | | |
| Loss-curve image | | |
| Untrained per-head heatmaps | | |
| Trained per-head heatmaps | | |
| Pairwise-distance results | | |
| Entropy results | | |
| Token-group attention results | | |
| Ablation results | | |
| Final predictions | | |

## 3. Stage A: NumPy Mechanics

### Expected versus actual shapes

| Variable | Expected shape | Actual shape | Verified? |
| :--- | :--- | :--- | :---: |
| $\mathbf{X}$ | $(T,d_{\text{model}})$ | | |
| $\mathbf{W}_Q^{(1)}$ | $(d_{\text{model}},d_k)$ | | |
| $\mathbf{W}_K^{(1)}$ | $(d_{\text{model}},d_k)$ | | |
| $\mathbf{W}_V^{(1)}$ | $(d_{\text{model}},d_v)$ | | |
| $\mathbf{Q}^{(1)}$ | $(T,d_k)$ | | |
| $\mathbf{K}^{(1)}$ | $(T,d_k)$ | | |
| $\mathbf{V}^{(1)}$ | $(T,d_v)$ | | |
| Head 1 scores | $(T,T)$ | | |
| $\mathbf{A}^{(1)}$ | $(T,T)$ | | |
| $\mathbf{H}^{(1)}$ | $(T,d_v)$ | | |
| $\mathbf{H}^{(2)}$ | $(T,d_v)$ | | |
| All head outputs | $(h,T,d_v)$ | | |
| All attention matrices | $(h,T,T)$ | | |
| Concatenated $\mathbf{C}$ | $(T,h d_v)$ | | |
| $\mathbf{W}_O$ | $(h d_v,d_{\text{model}})$ | | |
| Final $\mathbf{Y}$ | $(T,d_{\text{model}})$ | | |

### Attention row-sum check

Record the smallest and largest row sums observed for every head.

| Seed | Head | Minimum row sum | Maximum row sum | Approximately $1$? |
| :--- | ---: | ---: | ---: | :---: |
| | 1 | | | |
| | 2 | | | |
| | 3 | | | |
| | 4 | | | |

### Causal-mask check

| Check | Actual result | Verified? |
| :--- | :--- | :---: |
| Future-token weights are zero | | |
| First query has only one valid key | | |
| Mask is applied in every head | | |
| No row becomes all masked | | |

### Mechanics conclusion

What was verified about the forward pass?

>

What remains unverified because training has not occurred?

>

## 4. Stage B: Training Evidence

### Loss before and after training

Record held-out loss using the same evaluation procedure for every seed.

| Seed | Initial loss | Final training loss | Final held-out loss | Best epoch | Notes |
| ---: | ---: | ---: | ---: | ---: | :--- |
| | | | | | |
| | | | | | |
| | | | | | |

### Prediction check

| Seed | Input sequence | Target | Prediction before training | Prediction after training | Correct? |
| ---: | :--- | :--- | :--- | :--- | :---: |
| | | | | | |
| | | | | | |

Did the model learn more than a single memorized transition?

>

What held-out evidence supports the answer?

>

## 5. Stage C: Per-Head Visual Inspection

Complete one row for every saved heatmap.

| Seed | Example | Head | Prediction made before inspection | Pattern actually observed | Artifact path |
| ---: | :--- | ---: | :--- | :--- | :--- |
| | | 1 | | | |
| | | 2 | | | |
| | | 3 | | | |
| | | 4 | | | |

For each observation, answer:

1. Which query row was inspected?
2. Which key positions received the largest weights?
3. Was the pattern consistent across multiple examples?
4. Which parts of the heatmap were forced by causal masking?
5. What can the heatmap establish?
6. What can it not establish?

## 6. Stage D: Quantitative Head Comparison

### Pairwise mean absolute distance

Record one value for every head pair. State whether the values are from one
example or averaged across the held-out set.

| Seed | Evaluation scope | Head pair | Mean absolute distance | Interpretation |
| ---: | :--- | :--- | ---: | :--- |
| | | 1 and 2 | | |
| | | 1 and 3 | | |
| | | 1 and 4 | | |
| | | 2 and 3 | | |
| | | 2 and 4 | | |
| | | 3 and 4 | | |

Which pair had the most similar attention routing?

>

Does that prove functional redundancy? Why or why not?

>

### Normalized attention entropy

Use the number of valid keys in each query row when causal masking is active.

| Seed | Head | Mean normalized entropy | Most concentrated query | Most diffuse query |
| ---: | ---: | ---: | :--- | :--- |
| | 1 | | | |
| | 2 | | | |
| | 3 | | | |
| | 4 | | | |

What does entropy reveal?

>

What does entropy not reveal?

>

### Attention mass to a predefined business group

Define groups before inspecting the trained results.

| Group name | Token members | Why defined |
| :--- | :--- | :--- |
| Inventory-related | | |
| Financial | | |
| Operational | | |
| Other | | |

Record the measured mass for the selected query.

| Seed | Query token | Head | Group | Mean attention mass | Number of examples |
| ---: | :--- | ---: | :--- | ---: | ---: |
| | Forecast | 1 | | | |
| | Forecast | 2 | | | |
| | Forecast | 3 | | | |
| | Forecast | 4 | | | |

Did a pattern appear consistently enough to justify a cautious role
hypothesis?

>

## 7. Stage E: Head Ablation

Ablate one contextual head output at a time before concatenation. Keep all
trained parameters fixed and use the same held-out examples as the baseline.

| Seed | Condition | Held-out loss | $\Delta L$ from baseline | Prediction change | Interpretation |
| ---: | :--- | ---: | ---: | :--- | :--- |
| | All heads active | | $0$ | | |
| | Head 1 ablated | | | | |
| | Head 2 ablated | | | | |
| | Head 3 ablated | | | | |
| | Head 4 ablated | | | | |

Which head had the largest measured individual contribution?

>

Which head had the smallest measured individual contribution?

>

Does a small individual ablation delta prove that the head is universally
useless?

>

Were any pairs or groups of heads ablated together? If so, record the result.

>

## 8. Stability Across Seeds

Head numbers do not have fixed meanings across independent training runs.

| Observation | Seed 1 | Seed 2 | Seed 3 | Stable across runs? |
| :--- | :--- | :--- | :--- | :---: |
| Lowest held-out loss | | | | |
| Most concentrated head | | | | |
| Largest individual ablation delta | | | | |
| Strongest inventory-group mass | | | | |
| Closest pair of heads | | | | |

Which behavior repeated even when the head index changed?

>

Which interpretation was unstable?

>

## 9. Theory, Expectation, Observation, Intervention

| Evidence category | Recorded statement |
| :--- | :--- |
| What MHA theory permits | |
| What I expected before training | |
| What the untrained model produced | |
| What the trained heatmaps showed | |
| What numerical comparison showed | |
| What ablation changed | |
| My scoped conclusion | |
| What remains unproven | |

Rewrite any conclusion containing the words **always**, **proves**, or
**understands** unless the evidence genuinely supports that strength.

## 10. Errors, Surprises, and Limitations

### Implementation errors encountered

| Error | Cause | Evidence used to diagnose it | Fix |
| :--- | :--- | :--- | :--- |
| | | | |

### What surprised me?

>

### What contradicted my prediction?

>

### Experimental limitations

Record limitations such as:

- small dataset;
- few random seeds;
- short sequences;
- selected examples rather than complete evaluation;
- unstable training;
- interaction between heads not covered by single-head ablation;
- attention weights showing routing but not complete attribution.

>

## 11. Teach-Back and Mastery Check

Answer without notes:

1. Why can one attention head be limiting?
2. Why does every head receive the full sequence?
3. Why are several smaller heads not equivalent to one wider head?
4. Why is concatenation performed along the feature axis?
5. What does $\mathbf{W}_O$ learn?
6. What is the difference between attention-pattern similarity and functional
   redundancy?
7. What does entropy measure?
8. What does head ablation measure?
9. Why is one heatmap insufficient evidence of specialization?
10. How does returning to $d_{\text{model}}$ prepare the MHA output for a
    residual connection?

### Current confidence

| Capability | Confidence from 1 to 5 | Evidence |
| :--- | :---: | :--- |
| Explain the need for MHA | | |
| Trace every shape | | |
| Implement MHA from first principles | | |
| Debug axis and mask errors | | |
| Compare trained heads quantitatively | | |
| Perform and interpret ablation | | |
| State evidence boundaries | | |

## 12. Final Reflection

The strongest evidence that I understand Week 4 is:

>

The weakest part of my current understanding is:

>

The next experiment I would run is:

>

I am ready to move forward when:

>
