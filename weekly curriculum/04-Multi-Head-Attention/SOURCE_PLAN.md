# Week 4 Source Record

This file records the sources actually consulted for the Week 4 curriculum,
the role each source played, and the boundaries placed on the resulting
claims.

## Source Selection Rule

Use the smallest set of strong sources that covers:

- beginner intuition;
- the canonical Multi-Head Attention mechanism;
- mathematical and training clarification;
- evidence about specialization and redundancy;
- limits of interpreting attention weights.

Peer submissions and generated notes are not treated as technical authority.

## Local Sources Consulted

### Hands-on Large Language Models

[Local reference](../../resources/references/Hands-on-%20Large%20Language%20Models.md)

Used for:

- beginner intuition;
- visual Transformer architecture framing;
- the Week 3 to Week 4 bridge;
- clarification that heads run in parallel with distinct learned projections.

### Deep Learning

[Local reference](../../resources/references/Deep%20Learning.md)

Used for:

- objective, loss, gradient, and optimization terminology;
- the high-level explanation of how per-head parameters and the output
  projection learn through backpropagation.

This source was not used as the canonical definition of Multi-Head Attention.

## Primary External Sources Consulted

### Canonical mechanism

Vaswani et al. (2017),
[Attention Is All You Need](https://arxiv.org/abs/1706.03762), especially
Section 3.2.2.

Used for:

- the canonical Multi-Head Attention definition;
- independent projected Query, Key, and Value inputs per head;
- concatenation and output projection;
- the original equal-width convention
  $d_k=d_v=d_{\text{model}}/h$;
- the comparison with one full-width attention head.

### Head redundancy and pruning

Michel, Levy, and Neubig (2019),
[Are Sixteen Heads Really Better than One?](https://proceedings.neurips.cc/paper/2019/hash/2c601ad9d2ff9bc8b282670cdd54f69f-Abstract.html).

Used for:

- evidence that some trained heads can be removed with limited performance
  impact in particular evaluated models;
- the distinction between architectural capacity and observed head necessity;
- motivation for controlled head ablation.

The paper's findings are not generalized to every Transformer, task, layer, or
Week 4 training run.

### Head specialization

Voita et al. (2019),
[Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://aclanthology.org/P19-1580/).

Used for:

- evidence that some important heads can develop consistent,
  linguistically-interpretable patterns in specific machine-translation
  models;
- evidence that specialization and redundancy can coexist;
- the rule that role claims require measured behavior rather than
  architecture-based expectations.

The paper's linguistic head roles are not converted into assumed ForecastIQ
business roles.

### Attention interpretation

Jain and Wallace (2019),
[Attention is not Explanation](https://aclanthology.org/N19-1357/).

Wiegreffe and Pinter (2019),
[Attention is not not Explanation](https://aclanthology.org/D19-1002/).

Used together to establish a cautious boundary:

- attention weights are observable internal routing values;
- one heatmap is not automatically a complete causal explanation;
- interpretation depends on the question, model, baselines, and supporting
  tests;
- visualization should be combined with numerical comparison and
  intervention.

## Source Explicitly Excluded

The local file
resources/references/Build a Large Language Model (From Scratch).md
is not cited for Week 4.

The repository's reference inventory says that file contains the wrong
Raschka book content. No claim or implementation was derived from it.

## Implementation Evidence Rule

The NumPy formulas and examples are synthesized from the canonical mechanism
and the repository's established attention implementation conventions.

Every fixed numerical example must be recomputed before publication. A random
forward pass may demonstrate mechanics and shapes, but it cannot demonstrate
learned specialization.

Implementation output becomes evidence only after:

1. the model is trained;
2. the evaluation input and random seed are recorded;
3. actual attention matrices and losses are saved;
4. predictions are separated from observations;
5. ablation is evaluated against a baseline.

## Final Claim Boundaries

The Week 4 curriculum may state:

- every standard head receives the full input sequence;
- heads use different learned projections;
- one head produces one attention distribution per query;
- multiple heads permit multiple independently normalized distributions;
- the equal-width design is a common convention, not a universal definition;
- specialization may emerge during training;
- redundancy may also emerge;
- output projection mixes concatenated head features;
- attention analysis requires evidence from the trained model.

The curriculum must not state without measured evidence:

- that a specific head learned Inventory, Finance, Contracts, or Operations;
- that different-looking random heatmaps demonstrate specialization;
- that similar heatmaps prove functional redundancy;
- that more heads always improve the model;
- that a removable head is universally useless;
- that attention weights fully explain a prediction.

## Status

The research required for the Week 4 teaching material and permanent Topics
26-28 has been consulted and incorporated.

Research is no longer the blocking item. The remaining evidence must come from
the Week 4 implementation and its recorded experiments.
