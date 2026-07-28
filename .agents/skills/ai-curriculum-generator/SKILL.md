---

name: ai-curriculum-generator
description: >-
Transforms a raw technical assignment or problem statement into a structured,
first-principles AI learning module. Generates prerequisite maps, consolidated
study material, permanent concept notes, manual mathematical exercises,
implementation guidance, review material, and reflection artifacts while
following the pedagogical rules of the ai-learning-lab repository.
------------------------------------------------------------------

# First-Principles AI Curriculum Generator

## Overview

This skill orchestrates the creation of a new weekly curriculum module for the
`ai-learning-lab` repository.

Its purpose is not merely to generate notes or solve an assignment.

Its purpose is to design a learning path that allows the learner to understand:

* why each concept exists;
* what limitation from previous knowledge motivates it;
* how the mechanism works;
* the mathematics required to understand it;
* how the mathematics maps to code;
* how the concept connects to the current assignment;
* what limitations remain;
* what concept logically comes next.

The curriculum must remain beginner-first, dependency-driven, and
evidence-based.

Follow the teaching and learning rules defined in:

* `.agents/AGENTS.md`
* `LEARNING_RULES.md`
* `LEARNING_WORKFLOW.md`
* `resources/REFERENCE_MAP.md`

The learner is not assumed to be taking another AI or mathematics course.
Generated learning material must therefore be sufficient to teach the required
concepts from first principles.

---

# Core Principles

## 1. Learning Before Implementation

Do not begin by generating the final implementation.

The learner should first understand:

Problem
→ Need
→ Intuition
→ Mechanism
→ Mathematics
→ Worked Example
→ Math-to-Code Mapping
→ Implementation

Implementation should confirm understanding, not replace it.

---

## 2. Problem-First Teaching

Every NEW core concept must begin with the limitation that makes the concept
necessary.

Do not introduce a formal definition or formula before the learner understands
the problem being solved.

Preferred sequence:

Problem
→ Need
→ Intuition
→ Definition
→ Mechanism
→ Required Mathematics
→ Complete Worked Example
→ Math-to-Code Mapping
→ What-If Experiments
→ Misconceptions
→ Limitations
→ Assignment Connection
→ Modern AI Connection
→ Next Concept Bridge
→ Teach-Back
→ Quick Revision
→ My Understanding
→ Flashcards
→ Sources

---

## 3. Dependency-Driven Curriculum

Do not treat each week as an isolated lesson.

Every new week must identify:

* what the learner already knows;
* what concepts can be reused;
* what prerequisite gaps exist;
* what limitation of previous concepts motivates the new assignment;
* what concepts must be learned now;
* what concepts should deliberately be deferred.

Do not teach advanced concepts merely because they are related.

Teach only what is required for the current learning objective, plus enough
context to understand where the concept fits.

---

## 4. Best Reference for the Job

Use `resources/REFERENCE_MAP.md` to determine which reference should be
consulted for each concept.

Do not consult every available book for every topic.

Use the smallest number of references necessary.

Typical source roles may include:

* primary technical source;
* beginner intuition source;
* from-scratch implementation source;
* mathematical clarification source;
* optional backup source.

For foundational algorithms, architectures, or historical mechanisms, consult
the original paper when appropriate.

Peer submissions, team submissions, blog posts, or existing generated material
may be inspected for:

* additional questions;
* experiments;
* learner misconceptions;
* teaching ideas.

They are NOT authoritative sources of technical truth.

Never copy large sections from references.

Understand the material, then synthesize it in beginner-friendly language.

---

## 5. Evidence-Driven Learning

Never present an expected model behaviour as though it has already been
observed.

Always distinguish:

* what theory guarantees;
* what we expect may happen;
* what the actual implementation produced.

For example, do not say:

"The attention head focuses on authorization because it is important."

unless the measured attention weights actually show that behaviour.

Instead say:

"We might expect authorization to become relevant. Inspect the attention
weights and compare the observation with that expectation."

Unexpected results are learning opportunities.

---

# Workflow

## 1. Understand the Assignment

Read the assignment, exercise, problem statement, or requested learning goal.

Determine:

* what capability the assignment is intended to teach;
* what the learner is expected to build;
* what the learner should be able to explain afterward;
* what the learner should be able to calculate manually;
* what the learner should be able to implement independently;
* what limitation from the previous week naturally leads to this assignment.

Identify the central learning objective.

Do not generate curriculum files yet.

---

## 2. Inspect Existing Knowledge

Inspect relevant repository material before deciding what needs to be taught.

Review where relevant:

* existing `topics/` notes;
* previous weekly curriculum;
* previous assignments;
* learner reflections;
* learner review answers;
* learner experiments;
* `ROADMAP.md`;
* `PROGRESS.md`;
* `LEARNING_RULES.md`;
* `LEARNING_WORKFLOW.md`;
* `.agents/AGENTS.md`;
* `resources/REFERENCE_MAP.md`.

Identify three categories.

### Already Known

Concepts the learner has already studied sufficiently and can reuse.

Do not reteach these from scratch unless a clarification is necessary.

### Required Now

Concepts necessary to complete and understand the current assignment.

These form the current learning path.

### Deferred

Related concepts that are useful later but are not required now.

Explicitly defer them instead of expanding the current week unnecessarily.

Also identify:

* missing prerequisites;
* misconceptions that could block learning;
* prerequisite topics that require a small clarification;
* previously completed material that should remain untouched.

Do not mass-upgrade previous topics for formatting consistency.

---

## 3. Research the Concepts

For every NEW concept classified as Required Now:

1. Consult `resources/REFERENCE_MAP.md`.
2. Select the smallest appropriate set of references.
3. Prefer the original paper when the concept originates from a foundational
   algorithm or architecture.
4. Verify mathematical formulas, dimensions, terminology, and historical
   claims.
5. Distinguish standard definitions from implementation conventions.
6. Identify simplifications or assumptions used in beginner explanations.

Do not rely on generated notes as the sole source of technical truth.

Before teaching a formula, understand:

* what every symbol means;
* what assumptions the formula relies on;
* what the operation produces;
* why the operation is necessary;
* what shape each vector, matrix, or tensor has.

---

## 4. Design the Learning Path

Before creating large curriculum files, design the week.

Produce a proposed learning path containing:

* previous-topic bridge;
* central problem;
* assignment goal;
* learner capability at the end of the week;
* existing prerequisites;
* missing prerequisites;
* new concepts;
* dependency order;
* proposed permanent topic files;
* mathematical operations that require manual practice;
* concepts deliberately deferred;
* proposed implementation scope;
* proposed consolidated study-guide structure.

The dependency sequence should explain WHY each concept comes next.

Example:

Static Token Representation
→ Need Context
→ Sequence Representation
→ Pairwise Comparison
→ Query / Key / Value
→ Scaled Dot Product
→ Self-Attention
→ Causal Masking

Do not create a sequence merely because it matches textbook chapter order.

### Approval Gate

Present the proposed learning path to the learner before generating large new
curriculum material.

Do not proceed with large-scale curriculum generation until the learner
approves the structure.

Minor prerequisite clarifications do not require separate approval.

---

## 5. Create the Week Study Pack

After the learning path is approved, create the weekly curriculum folder.

Preferred naming convention:

`weekly curriculum/<week-number>-<topic-name>/`

Create:

* `PREREQUISITE_MAP.md`
* `PREREQUISITE_KNOWLEDGE.md`
* `Week X Topics in Detail.md`
* `REVIEW.md`
* `REFLECTION.md`

Additional files may be created only when they serve a clear learning purpose.

### PREREQUISITE_MAP.md

Create a dependency diagram showing:

* prerequisite concepts from earlier weeks;
* new concepts introduced this week;
* dependency arrows;
* important limitations that motivate later concepts;
* concepts deferred to future weeks when useful.

Prefer Mermaid diagrams when appropriate.

The map answers:

"What must I understand, and in what order?"

---

### PREREQUISITE_KNOWLEDGE.md

Create one consolidated beginner-friendly study guide for the week.

It must cover every concept classified as Required Now.

The guide should tell one continuous learning story rather than duplicating
every permanent topic note in full.

It must include:

* previous-week bridge;
* central problem;
* intuition;
* definitions;
* required mathematics;
* shape transformations;
* at least one complete numerical worked example;
* assignment connection;
* important misconceptions;
* limitations;
* links to permanent topic notes.

The guide must be sufficient for a beginner who has no external course.

---

### Week X Topics in Detail.md

List the permanent topic notes in dependency order.

For each topic, state briefly:

* what it teaches;
* why it appears at that point in the sequence.

Do not merely provide filenames.

---

### REVIEW.md

Review mastery across four dimensions:

1. Conceptual understanding
2. Mathematics and shape tracing
3. Code comprehension / implementation
4. Debugging and what-if reasoning

Questions should test understanding, not memorization.

Do not ask concepts that were deliberately deferred.

---

### REFLECTION.md

Create a structure where the learner records evidence from their implementation.

Where relevant, include:

* expected shapes;
* actual shapes;
* expected behaviour;
* observed behaviour;
* numerical outputs;
* experiments;
* interpretation;
* differences between prediction and observation.

---

## 6. Create New Permanent Topics

Create permanent notes only for NEW core concepts required by the current week.

Store them in:

`topics/`

Continue the repository's numeric dependency-order naming convention.

Example:

`19 - LM - Context Windows and Sequence Representations.md`

Do not regenerate earlier topics merely to make them match newer formatting.

Update an existing topic only when:

* a factual error is discovered;
* a misleading statement blocks the current topic;
* a small prerequisite clarification is required;
* the learner explicitly revisits the topic.

Make the smallest useful change.

### Required Permanent Topic Structure

When applicable, each new topic should contain:

1. The Problem
2. Why We Need Something New
3. One-Line Definition
4. Beginner Intuition / Mental Model
5. What Came Before → What Changes Now
6. How It Works
7. Required Mathematics
8. Complete Worked Example
9. Math → Code Mapping
10. Experiments / What-If Questions
11. Common Misunderstandings
12. Limitations and Trade-Offs
13. Where It Appears in the Current Assignment
14. Where It Appears in Modern AI Systems
15. Connection to the Next Concept
16. Teach-Back and Small Application Exercise
17. Quick Revision Summary
18. My Understanding
19. Flashcards
20. Sources

Do not create empty sections purely to satisfy structure.

If a section is not useful for the concept, adapt intelligently.

---

# Mathematics Teaching Rules

When a formula is introduced, do not only display the equation.

Explain every component.

For example, for:

`Attention(Q, K, V) = Softmax((QKᵀ) / √d_k)V`

explain individually:

* `Q`;
* `K`;
* `V`;
* `Kᵀ`;
* `QKᵀ`;
* `d_k`;
* `√d_k`;
* division by `√d_k`;
* Softmax;
* multiplication by `V`;
* output shape.

For matrix or tensor operations:

* show input shapes;
* explain why multiplication is valid;
* show output shape;
* explain what one output cell means when useful.

Use small numerical examples before large realistic dimensions.

Do not hide assumptions.

If a statement such as:

`Var(q · k) = d_k`

depends on assumptions, state those assumptions.

---

## 7. Create Manual Exercises

Create manual exercises only for mathematics or mechanisms that materially
benefit from hand calculation.

Store them under:

`manual-exercises/`

inside the weekly curriculum folder.

For each exercise:

* use deliberately small numbers;
* provide every required vector or matrix;
* state all dimensions;
* provide learner workspace;
* require intermediate calculations;
* provide the solution only after the workspace;
* verify every expected numerical result;
* connect the exercise directly to the current assignment.

The learner should calculate important mechanisms manually at least once before
relying on library abstractions.

Examples that may justify manual exercises:

* embedding lookup;
* matrix multiplication;
* cosine similarity;
* Softmax;
* attention score calculation;
* masking;
* weighted sums.

Do not create manual exercises merely to satisfy structure.

---

## 8. Build the Implementation

Create implementation material only after the required learning material and
manual prerequisites are complete.

Prefer building the core mechanism from first principles when practical.

The implementation must:

* start with small, inspectable examples;
* expose intermediate values;
* print or inspect important shapes;
* map code operations to mathematical operations;
* include explanatory Markdown around code;
* avoid monolithic final solution blocks;
* encourage the learner to implement parts independently;
* distinguish theory, expected behaviour, and observed output.

For notebooks:

* use Markdown cells to explain every meaningful stage;
* separate conceptual stages into manageable code cells;
* do not provide a giant finished implementation with little explanation;
* add new third-party dependencies to `requirements.txt` when required.

The learner should understand each stage before moving to the next.

---

## 9. Experiment and Review

After the baseline implementation works, require experimentation.

Change at least one meaningful variable, assumption, or configuration.

For each experiment:

1. State the change.
2. Predict what may happen.
3. Run the experiment.
4. Record the actual output.
5. Compare expectation with observation.
6. Explain the result.
7. Identify a limitation or new question exposed by the experiment.

Useful experiments may include:

* changing initialization;
* changing embedding dimension;
* changing sequence order;
* removing scaling;
* removing masking;
* changing context length;
* varying number of heads;
* altering an input token.

Do not rewrite observations to match expectations.

Review:

* conceptual understanding;
* mathematical understanding;
* code comprehension;
* debugging ability;
* what-if reasoning.

---

## 10. Reflection and Mastery

A week is complete only when the learner can:

* explain why the central concept exists;
* explain the problem it solves;
* explain the mechanism in plain language;
* trace important mathematics;
* trace matrix/tensor shapes;
* implement the core mechanism;
* explain implementation outputs;
* distinguish expectation from observation;
* identify limitations;
* explain how the concept connects to previous topics;
* explain what unresolved problem motivates the next topic.

Update where appropriate:

* `REVIEW.md`
* `REFLECTION.md`
* `PROGRESS.md`
* mastery/confidence tracking
* flashcards

Do not mark a topic complete merely because the code runs.

---

# Beginner-First Explanation Rules

The learner must not be expected to define an unfamiliar concept before being
taught.

Do not use:

Undefined Question
→ Learner Guesses
→ Agent Corrects Guess

Use:

Teach
→ Demonstrate
→ Clarify
→ Check Understanding
→ Apply
→ Reflect

When asking teach-back questions:

* ask one question at a time;
* start simple;
* move toward practical application;
* correct misunderstandings immediately;
* explain WHY the answer is incorrect;
* do not simply provide the correct answer.

---

# Analogy Rules

Analogies are encouraged when they genuinely improve intuition.

For each important analogy:

* map parts of the analogy to the technical concept;
* explain what the analogy helps illustrate;
* explain where the analogy stops being accurate when necessary.

Do not allow an analogy to replace the technical mechanism.

---

# Visual Explanation Rules

Use diagrams when spatial, architectural, or flow relationships are difficult
to understand from prose alone.

Useful cases include:

* matrix shapes;
* neural-network flow;
* embedding spaces;
* attention matrices;
* Transformer architecture;
* dependency relationships.

Do not generate decorative visuals that add no learning value.

---

# Source Rules

Every permanent topic should identify its important sources.

Prefer:

1. original paper for the canonical mechanism;
2. mapped primary reference;
3. mapped implementation reference;
4. optional clarification reference when necessary.

Do not cite a source that was not actually consulted.

Do not use peer/team submissions as technical authority.

Do not over-reference.

Two strong references are preferable to six redundant ones.

---

# Scope Control

Avoid curriculum expansion.

If the learner needs:

A → B → C

to complete the assignment, do not automatically teach:

D → E → F

because those concepts happen to appear in the same textbook chapter.

Record useful future concepts as deferred.

Examples:

* positional encoding;
* residual connections;
* LayerNorm;
* FeedForward networks;
* grouped-query attention;
* KV caching;
* FlashAttention;
* inference optimisation.

Introduce them only when they become necessary.

---

# Legacy Topic Preservation

The enhanced permanent-topic format applies prospectively.

Do not mass-rewrite completed Week 1 or Week 2 material simply to make formatting
consistent.

Existing learning material represents completed learning history.

Modify previous material only when:

* a factual error is discovered;
* a misleading explanation blocks current learning;
* the learner revisits the concept;
* a small clarification materially improves the current prerequisite chain.

Preserve correct content wherever possible.

---

# Common Mistakes to Avoid

## Generating Before Designing

Do not immediately create files after reading an assignment.

Understand the learning objective and design the dependency path first.

---

## Reteaching Everything

Do not repeat concepts the learner already understands.

Reuse previous topics and link back to them.

---

## Overloading the Week

Do not introduce advanced concepts that are not necessary for the current
assignment.

---

## Treating Generated Content as Truth

Verify formulas, shapes, terminology, architectural claims, and numerical
examples against authoritative references.

---

## Skipping Mathematical Verification

Never include hand-calculated expected outputs without independently verifying
them.

---

## Hiding Assumptions

Do not present conditional mathematical results as universal facts.

State simplifying assumptions explicitly.

---

## Confusing Convention with Requirement

Distinguish mathematical definitions from common implementation conventions.

For example:

`d_k = d_model / h`

may be a standard design choice in a particular Multi-Head Attention
implementation, but it should not automatically be presented as the universal
definition of Multi-Head Attention.

---

## Giving the Final Solution Too Early

The implementation is a learning exercise.

Do not turn the notebook into an answer key that removes the learner's need to
reason.

---

## Monolithic Notebooks

Avoid huge code cells with little explanation.

Break the implementation into conceptual stages.

---

## Assuming Experimental Behaviour

Do not claim a model learned a relationship solely because theory suggests it
might.

Inspect actual outputs.

---

## Mass-Upgrading Legacy Material

Do not rewrite previous weeks simply because the new curriculum format is
better.

Keep completed learning history stable.

---

# Completion Standard

The curriculum generator has completed its job only when the resulting module
provides a clear path from:

Previous Knowledge
→ Visible Limitation
→ New Concept
→ Intuition
→ Mathematics
→ Worked Example
→ Manual Practice
→ Implementation
→ Experiment
→ Reflection
→ Mastery
→ Next Concept

The final goal is not to generate more files.

The final goal is to make the learner capable of explaining, calculating,
implementing, testing, and reasoning about the concept independently.


Reference Resolution Rule

1. Read resources/REFERENCE_MAP.md to identify preferred sources.
2. Read resources/LOCAL_REFERENCE_LIBRARY.md to determine which sources
   are actually available locally.
3. Prefer an available local reference when it is appropriate for the topic.
4. If a mapped reference is not locally available:
   - do NOT pretend to have read it;
   - do NOT invent chapter numbers, quotes, or claims from it;
   - if web/search grounding is available, locate an authoritative external
     source instead;
   - prefer original papers, official documentation, official author material,
     or open authoritative textbooks.
5. Record which sources were actually consulted in the generated topic.
6. If neither the local reference nor an authoritative external source can be
   accessed, state that the preferred reference was unavailable and proceed
   only with claims that can be independently supported.