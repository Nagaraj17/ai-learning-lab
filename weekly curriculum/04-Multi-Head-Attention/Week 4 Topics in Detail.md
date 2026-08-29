# Week 4 Topics in Detail: Multi-Head Attention

Week 4 is the continuation of Week 3, not a fresh start. Week 3 taught how one
attention head can produce contextual representations. Week 4 begins by asking
what one head cannot do well enough on its own.

The permanent topic notes for this week should be read in the order below.

1. **[26 - TRANSFORMER - Multi-Head Attention](../../topics/26%20-%20TRANSFORMER%20-%20Multi-Head%20Attention.md)**
   - Introduces the visible limitation of a single learned attention
     perspective and explains why multiple independently projected heads may
     help.
   - This note comes first because the learner must understand the problem
     before learning concatenation or output projection.

2. **[27 - TRANSFORMER - Concatenation and Output Projection](../../topics/27%20-%20TRANSFORMER%20-%20Concatenation%20and%20Output%20Projection.md)**
   - Explains how per-head outputs are combined, why simple side-by-side
     stacking is not enough, and how `W_O` mixes the joined features back into
     model space.
   - This note appears second because it depends on already understanding what
     each head produces.

3. **[28 - TRANSFORMER - Inspecting Specialization and Redundancy in Attention Heads](../../topics/28%20-%20TRANSFORMER%20-%20Inspecting%20Specialization%20and%20Redundancy%20in%20Attention%20Heads.md)**
   - Distinguishes theory from evidence by examining what trained heads
     actually do, when heads become redundant, and what head ablation can tell
     us.
   - This note appears third because evidence about specialization only makes
     sense after the learner understands the mechanics of per-head attention
     and output combination.

## Week 4 Sequence Inside the Study Pack

The weekly curriculum will follow this order:

1. Week 3 bridge: embeddings to single-head attention
2. What one head currently does
3. Why one learned perspective can be limiting
4. Same sequence through multiple heads
5. Independent per-head projections
6. Head dimensions and shape tracing
7. Concatenation
8. Output projection `W_O`
9. Full forward pass
10. Per-head visual inspection
11. Training, specialization, redundancy, and ablation
12. Preview of residual connections and LayerNorm
13. Bridge to the later Transformer block week

## Week 3 Boundary Correction

Multi-Head Attention is no longer treated as a Week 3 stretch topic in the
curriculum sequence. Week 3 remains the single-head attention week. Week 4 is
the first week where Multi-Head Attention is taught in full.
