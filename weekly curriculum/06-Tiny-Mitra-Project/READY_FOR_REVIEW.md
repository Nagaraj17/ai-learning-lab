# Tiny Mitra — ready for your next session

Prepared October 18 for your October 19 login. The code, trained checkpoints and presentation are ready for review. **Your learning and rehearsal remain open.** You do not need to read every generated file first.

## Where to begin

1. Open [Your first session](../../projects/Week%206/tiny-mitra/START_HERE.md). Run the small `Jo / DrugA` teaching example, then change DrugA to DrugB.
2. Read the source/target and tokenization sections in [the refresher](PREREQUISITE_KNOWLEDGE.md). We will then walk through embeddings and position together, followed by one attention prediction.
3. Use [the presentation](../../presentations/mitra/output/Tiny_Mitra_Final_v7.pptx) and [detailed speaking guide](../../presentations/mitra/output/Mitra_Final_Presenter_Guide.md) as a map, not a script to memorize. It has 28 core slides and eight appendix slides.
4. Read [the final findings](../../projects/Week%206/tiny-mitra/reports/FINAL_FINDINGS.md) after the mechanism makes sense. Every finding states the change, observation, interpretation and limit.

## What is available

- A two-block GPT assembled manually from allowed PyTorch primitives, plus seven separately trained architecture variants; eight primary and eight additional seed runs completed on one frozen synthetic corpus.
- Gradio model selection, same-note comparison, actual incremental token generation, attention/head interventions and component evidence. All 28 checkpoint routes pass API verification; the live VISITED example emitted 12 matching token updates.
- Raw outputs and audited metrics for all 32 primary final model/partition rows and four separately reported demo rows. Training histories, teacher-forced loss/accuracy, precision/recall/F1, preservation, EOS, head/category findings and CPU timings are retained.
- A local model/source/results ZIP, a smaller Colab source bundle, two notebooks, comments, README and a recorded-output HTML fallback. The ZIP has a separate fresh-extraction verification report; see the handoff for its final status.

## What the model learned — and where it fails

The selected full demo is `final_seed7/transformer_2`. It passes all 11 authored development prompts, including the unseen Olivia/MRN example. It gives 49/50 exact notes with familiar names/layouts, and 43/50 with new names/familiar layouts. **It gives only 5/50 in either new-layout condition; those five successes are no-identifier controls.** This is a learning demonstration with measured transfer limits, not general de-identification reliability.

The primary architecture comparison remains seed 42. Its full model passes 10/11 authored notes; the PHONE failure remains visible. We do not substitute the selected demo's results into that benchmark. The smaller/no-residual variants sometimes outperform the full model; the presentation explains the actual evidence instead of assuming every added component helped.

## What needs us together

Confirm team members and the exact speaking slot, complete each presenter's prompt walkthrough, rehearse timing/questions, and check the latest UI visually. The latest automated browser recheck was blocked after a connection-refused page; API checks pass and earlier browser screenshots are saved. Also open the deck in the actual presentation application/projector setup. We can guide your first Colab sign-in and GPU/resume/CPU portability check when you are present; no actual cloud run has been claimed and no paid resources were purchased.

Use [INPUTS_NEEDED.md](INPUTS_NEEDED.md) and [TASKS.md](TASKS.md) for the remaining gates. All reviews so far are author self-reviews. Learning milestones stay open until you can explain and apply the concepts.
