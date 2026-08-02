# REFLECTION (Week 4: Multi-Head Attention)

Use this file to record actual evidence from the Week 4 implementation.

## 1. Stage A - NumPy Mechanics Evidence

### Shape Verification Table

| Variable | Expected Shape | Actual Shape | Verified |
|---|---|---|---|
| `X` | `(T, d_model)` | | |
| `Q_1` | `(T, d_k)` | | |
| `K_1` | `(T, d_k)` | | |
| `V_1` | `(T, d_v)` | | |
| `scores_1` | `(T, T)` | | |
| `attn_1` | `(T, T)` | | |
| `head_1` | `(T, d_v)` | | |
| `head_2` | `(T, d_v)` | | |
| `concat` | `(T, h * d_v)` | | |
| `output` | `(T, d_model)` | | |

### Row-Sum Check

Record whether each attention row sums to approximately `1.0`.

| Head | Row sums | Verified |
|---|---|---|
| Head 1 | | |
| Head 2 | | |

### Visual Inspection Notes

For each saved heatmap, record:

- what you expected
- what you observed
- what you can conclude
- what you cannot conclude

## 2. Stage B - Trainable Experiment Evidence

### Before vs After Training

| Seed | Loss Before Training | Loss After Training | Observation |
|---|---|---|---|
| | | | |
| | | | |

### Head Comparison Notes

| Seed | Head | Pattern observed | Clear specialization? | Redundancy signs? |
|---|---|---|---|---|
| | 1 | | | |
| | 2 | | | |
| | 1 | | | |
| | 2 | | | |

### Head Ablation

| Seed | Baseline loss | Loss with head 1 removed | Loss with head 2 removed | Interpretation |
|---|---|---|---|---|
| | | | | |
| | | | | |

## 3. Theory vs Expectation vs Observation

| Category | Write your evidence |
|---|---|
| What theory permits | |
| What we expected before training | |
| What the trained model actually produced | |

## 4. What Surprised Me?

- 

## 5. What Still Feels Unclear?

- 

## 6. Can I Teach This Now?

Try answering these without notes:

1. Why can one head be limiting?
2. Why do all heads still need the full sequence?
3. Why is concatenation alone not enough?
4. Why can ablation reveal redundancy?
