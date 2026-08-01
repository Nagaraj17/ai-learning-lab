# Week 4 Source Plan

This plan records which sources were checked during Phase 1 and which sources
must be consulted before Phase 2 and Phase 3 content is finalized.

## Source Selection Rule

Use the smallest set of strong sources that covers:

- beginner intuition
- canonical mechanism
- mathematical clarification
- implementation clarification
- evidence about specialization and redundancy

## Sources Already Verified as Locally Available

1. `resources/references/Hands-on- Large Language Models.md`
   - Role:
     - beginner intuition
     - visual architecture framing
     - Week 3 to Week 4 bridge language
   - Reason selected:
     - this is the mapped local reference for the Transformer curriculum

2. `resources/references/Deep Learning.md`
   - Role:
     - mathematical clarification
     - learning / backpropagation language when discussing how heads are
       trained through loss and updates
   - Reason selected:
     - this is the mapped local mathematics reference

## Sources Explicitly Not Safe to Cite for This Topic

1. `resources/references/Build a Large Language Model (From Scratch).md`
   - Status:
     - file exists, but the local reference inventory says it contains the
       wrong Raschka book content
   - Rule:
     - do not cite it for Multi-Head Attention

## External Sources Required Before Teaching Content Is Finalized

These are not treated as consulted yet. They must be read before the permanent
topic notes and experiments cite them.

1. Vaswani et al., *Attention Is All You Need*
   - Role:
     - canonical definition of Multi-Head Attention
     - equal-width head convention `d_k = d_model / h`
     - cost comparison framing

2. Michel, Levy, and Neubig, *Are Sixteen Heads Really Better than One?*
   - Role:
     - evidence about head redundancy and pruning

3. Voita et al., *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned*
   - Role:
     - evidence about specialization emerging unevenly across heads

4. Authoritative implementation clarification source for from-scratch
   Multi-Head Attention
   - Preferred option:
     - official author or original educational material from Sebastian Raschka
   - Fallback:
     - do not cite Raschka unless the actual correct source is located and
       consulted

## Claim Boundaries for Week 4

Until the external sources above are consulted, the Week 4 curriculum may
safely assert only the following:

- every head processes the full input sequence
- heads use different learned projections
- the common equal-width design is a convention, not the universal definition
- specialization may emerge during training
- redundancy can occur
- a random forward pass demonstrates mechanics, not learned behaviour

The curriculum must not claim:

- that any specific head learned a business role without measured evidence
- that more heads always help
- that attention weights fully explain model decisions
