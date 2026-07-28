# REFLECTION (Week 3: Attention)

Document your implementation outputs, shape verifications, and experimental evidence below:

## 1. Shape Verification Evidence Table

Paste the printed shapes from your NumPy implementation to prove end-to-end dimension correctness:

| Matrix Variable | Expected Shape Formula | Numerical Shape (T=3, d=2) | Verified (OK / XX) |
| :--- | :--- | :--- | :--- |
| Sequence Matrix $\mathbf{X}$ | $(T \times d_{model})$ | `(3, 2)` | |
| Query Matrix $\mathbf{Q}$ | $(T \times d_k)$ | `(3, 2)` | |
| Key Matrix $\mathbf{K}$ | $(T \times d_k)$ | `(3, 2)` | |
| Value Matrix $\mathbf{V}$ | $(T \times d_v)$ | `(3, 2)` | |
| Transposed Keys $\mathbf{K}^\top$ | $(d_k \times T)$ | `(2, 3)` | |
| Score Matrix $\mathbf{S}$ | $(T \times T)$ | `(3, 3)` | |
| Masked Scores $\mathbf{S}_{\text{masked}}$ | $(T \times T)$ | `(3, 3)` | |
| Attention Weights $\mathbf{A}$ | $(T \times T)$ | `(3, 3)` | |
| Contextual Output $\mathbf{H}$ | $(T \times d_v)$ | `(3, 2)` | |

## 2. Causal Masking Proof
Paste your printed $3 \times 3$ Attention Weight Matrix $\mathbf{A}$ below and verify that the upper triangle entries $A_{0,1}, A_{0,2}, A_{1,2}$ are strictly `0.0`:

```text
[Paste your numpy printed matrix A here]
```

- [ ] Verified: Token 1 (`"Order"`) has $100\%$ self-attention and $0\%$ future attention.
- [ ] Verified: Row sums of $\mathbf{A}$ all equal $1.000$.

## 3. Evidence-Driven Observations
Record observed model outputs when testing different input prompts. Distinguish clearly between:
- What theory allows.
- What we expected.
- What our actual code output produced.
