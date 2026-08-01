# Week 4 Phase Plan

Week 4 stays focused on one central question:

> A single attention head can build one contextual view of the full sequence.
> What happens when we let multiple independently parameterized heads examine
> that same sequence in parallel?

This week does **not** expand into the full Transformer block. Residual
connections and Layer Normalization appear only as short previews near the end.

## Phase 1 - Structure and Scope Alignment

Deliverables:

- Create the Week 4 curriculum folder.
- Define the Week 4 dependency map.
- Define the Week 4 topic sequence.
- Define the source plan.
- Make the smallest Week 3 scope corrections needed to move
  Multi-Head Attention fully into Week 4.

Success criteria:

- Week 3 ends at single-head self-attention plus causal masking.
- Week 4 begins from the visible limitation of one head.
- Links and Mermaid blocks validate.

## Phase 2 - Teaching Material

Deliverables:

- `PREREQUISITE_KNOWLEDGE.md`
- `Week 4 Topics in Detail.md`
- Permanent topic notes for:
  - `26 - TRANSFORMER - Multi-Head Attention.md`
  - `27 - TRANSFORMER - Concatenation and Output Projection.md`
  - `28 - TRANSFORMER - Inspecting Specialization and Redundancy in Attention Heads.md`
- `CHEAT_SHEET.md`

Teaching guardrails:

- Begin from the Week 3 bridge, not from the condensed MHA formula.
- Explain every symbol before using the full formula.
- Use the continuous Healthcare GPO / ForecastIQ sequence throughout.
- Distinguish theory, expectation, and measured experimental evidence.

## Phase 3 - Mathematics and Implementation

Deliverables:

- Manual exercises:
  - `manual-exercises/01-two-head-shape-tracing.md`
  - `manual-exercises/02-concatenation-and-output-projection.md`
- Progressive NumPy mechanics implementation in `projects/Week 4/`
- Heatmaps generated from actual outputs
- Small trainable experiment for evidence about specialization, redundancy,
  and head ablation

Implementation order:

1. Reconstruct a transparent single-head baseline.
2. Run two heads with independent weights.
3. Inspect per-head outputs and per-head attention matrices.
4. Concatenate the head outputs.
5. Apply `W_O`.
6. Add shape assertions and saved visual outputs.
7. Add a separate small trainable experiment.

## Phase 4 - Review and Mastery

Deliverables:

- `REVIEW.md`
- `REFLECTION.md`
- `PROGRESS.md` update showing Week 4 prepared or started
- Final validation pass

Mastery target for this week:

- Explain why one head can be limiting.
- Trace multi-head shapes without guessing.
- Implement the forward pass from first principles.
- Interpret head visualizations cautiously.
- Explain why specialization can emerge but is not guaranteed.
- Connect Multi-Head Attention to the later Transformer block topic.
