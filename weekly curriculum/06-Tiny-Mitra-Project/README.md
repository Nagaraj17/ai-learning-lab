# Tiny Mitra: start here

We are preparing a manually assembled small GPT and a local Gradio comparison demo for **Friday, October 23, 2026**.

The [implementation plan](IMPLEMENTATION_PLAN.md) records the learning sequence, PyTorch boundary, proposed project files, architectures, training/checkpoints, evaluation, hypotheses, head/component experiments, Gradio views and completion checks.

Use the [task ledger](TASKS.md) for current progress, acceptance criteria and review records. The [training environment guide](TRAINING_ENVIRONMENT.md) explains the recommended Colab training → downloaded checkpoints → local Gradio workflow, including alternatives and what remains unverified.

Start with Sections 1–4 of the plan. The remaining sections are a reference for implementation. We will review our own Weeks 1–5 examples as we reach each concept.

Our first learning session follows one example:

```text
PATIENT John DIAGNOSIS NSCLC
PATIENT [NAME] DIAGNOSIS NSCLC
```

We first clarify source versus target, tokenizer versus vocabulary, token IDs, embeddings and positional information. Then we agree on how the implementation represents unseen names and numbers.

The [refresher](PREREQUISITE_KNOWLEDGE.md), [Python project](../../projects/Week%206/tiny-mitra/README.md) and trained local demo are prepared. All eight final architectures and eight additional seed runs completed; both local environments pass34 tests, and all28 UI checkpoint routes pass API checks. Complete final outputs are reviewed, including severe new-layout failures. The36-slide presentation has detailed notes. Start with [READY_FOR_REVIEW.md](READY_FOR_REVIEW.md) for the short reading order and remaining gates. Every presenter's explanation, timed rehearsal, latest browser visual check and actual Colab verification remain open.

See [inputs needed](INPUTS_NEEDED.md) for the Colab/account, presentation and learner gates, and [execution handoff](HANDOFF.md) for the running work queue. Creating files or passing code checks does not complete learning milestones.
