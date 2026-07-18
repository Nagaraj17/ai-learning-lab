# Level 2: Embedding Lookup vs Matrix Multiplication

This exercise proves mathematically that passing a one-hot encoded vector through a Linear Layer ($x \cdot W$) is exactly identical to using the Token ID to perform an array index lookup ($W[TokenID]$).

## Initialization
- **Vocabulary Size:** 4
- **Embedding Dimension:** 3
- **Token ID for "Shipment":** `2`

**The Weight / Embedding Matrix ($W_{embed}$):**
```text
Row 0: [ 0.1, -0.2,  0.5]
Row 1: [ 0.9,  0.1, -0.1]
Row 2: [-0.4,  0.7,  0.2]
Row 3: [ 0.0,  0.8, -0.5]
```

---

## 1. The Week 1 Way: Matrix Multiplication
First, convert the Token ID (`2`) to a One-Hot vector.
- $x = [0, 0, 1, 0]$ (Shape `(1, 4)`)

Now multiply $x \cdot W_{embed}$.
- Column 0: $(0 \times 0.1) + (0 \times 0.9) + (1 \times -0.4) + (0 \times 0.0) = -0.4$
- Column 1: $(0 \times -0.2) + (0 \times 0.1) + (1 \times 0.7) + (0 \times 0.8) = 0.7$
- Column 2: $(0 \times 0.5) + (0 \times -0.1) + (1 \times 0.2) + (0 \times -0.5) = 0.2$
- **Result:** `[-0.4, 0.7, 0.2]`

## 2. The Week 2 Way: Array Index Lookup
Instead of multiplying, just use the Token ID (`2`) to ask the array for Row 2.
- `embedding = W_embed[2]`
- **Result:** `[-0.4, 0.7, 0.2]`

**Conclusion:** The result is exactly the same, but the Index Lookup required 0 math operations instead of 12!

---

## Learner Workspace

### Predict the Output
If our Token ID is `0`, what will the embedding vector be based on the matrix above?

### Blank Calculation Table
Try calculating a lookup vs multiplication manually.
- Token ID: `1`
- Matrix $W$:
  `[[0.5, 0.5], [0.8, -0.8], [0.1, 0.2]]`

| Method | Formula | Your Calculation | Expected Output |
|---|---|---|---|
| One-Hot | `one_hot(1)` | | |
| Multiplication | $x \cdot W$ | | |
| Array Index | `W[1]` | | |
