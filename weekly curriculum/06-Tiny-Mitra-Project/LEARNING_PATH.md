# Guided refresher and code walkthrough

**Saved resume point — October 19, 2026:** paused at Nagar's request. Begin next time at **Section 5: Position** in PREREQUISITE_KNOWLEDGE.md. Source/target, vocabulary/tokenization, token positions, shapes and embeddings have been discussed. In the embedding discussion, clarified that this project's vocabulary is declared in advance and that the sequence matrix is a lookup result; the shared embedding table contains learned parameters updated by the optimizer using gradients. No hands-on or full mastery gate is closed from discussion alone. L01 remains IN PROGRESS.

Nagar wants to understand the mechanisms first, then run training himself with guidance. Resume the lesson when he returns; no training or automatic lesson continuation is scheduled by this pause.

## Our starting point

Use the small **learning pilot** for the first walkthrough. It is a complete two-block GPT trained on four short examples. Its ordinary text uses character tokens; placeholders such as `[NAME]` remain single tokens. Later experiments use a different tokenizer and larger dimensions, so their settings must stay labelled separately.

| Setting | Learning pilot we are studying | Final comparison, for later |
| --- | ---: | ---: |
| Vocabulary entries | 109 | 146 |
| Numbers in each token vector | 32 | 64 |
| Attention heads per block | 4 | 4 |
| Decoder blocks | 2 | 2 |
| FFN dimensions | 32 → 96 → 32 | 64 → 192 → 64 |
| Available sequence positions | 128 | 256 |

These are saved implementation settings, not numbers to memorize. The refresher often uses the final model's 64-feature example; its mechanism also applies to the pilot's 32 features. We will name the setting whenever tracing a shape.

## Reading and walkthrough order

Each step follows: explain the problem → work one example → explain the necessary arithmetic → locate the matching code → let Nagar inspect/change/run a small example → clarify → check understanding. Ask one comprehension question at a time after teaching and reading. Leave My Understanding for Nagar.

| Step | Read in PREREQUISITE_KNOWLEDGE.md | Code we will open together | What we will understand or do |
| --- | --- | --- | --- |
| 1 | Sections 1–2 | `tools/pilot.py`, then `tokenizer.py` | Source versus expected target; characters, special tokens, vocabulary, IDs, encoding and decoding |
| 2 | Sections 3–5 | `model.py`: the embedding lookups at the start of `TinyGPT.forward` | Shapes, learned token vectors, learned position vectors and their addition |
| 3 | Section 6 and the softmax explanation in 7 | `model.py`: `nn.Linear` projections; `generate.py`: softmax/selection | Learned weights, bias, raw scores and probabilities |
| 4 | Sections 7–9 | `model.py`: `CausalAttention.forward` | Q/K/V, scaling, masks, value mixing, four heads and merging |
| 5 | Sections 10–11 | `model.py`: `LayerNorm`, `DecoderBlock`, then the full forward path | Residual additions, normalization, FFN and two complete blocks |
| 6 | Sections 12–13 | `data.py`: `training_item`; `train.py`: `train_run` | Training sequence, shifted targets, scored output/EOS positions, loss, backward and optimizer updates |
| 7 | Section 14 | `tools/pilot.py`, `train.py`, `generate.py` | Nagar starts a separate practice run, watches training, saves/reloads weights and generates a continuation |
| 8 | Sections 15–16 | `app.py`, then the evaluation/experiment files | How the screen calls a saved model; what comparisons and head interventions actually measure |

The code paths above describe the character pilot. The final task-piece model uses `task_tokenizer.py` and `task_runtime.py`; we will map those differences after understanding the shared path. We will then connect the walkthrough to our own Weeks 1–5 assignments and the presentation.

## First lesson: the training pair

```text
Source:          PATIENT Jo TREATMENT DrugA
Expected target: PATIENT [NAME] TREATMENT DrugA
```

The source is the note supplied to the model. The expected target is the example answer that teaches the synthetic policy: replace the patient's identity and preserve the treatment. During training, we compare next-token predictions with the correct continuation and update learned parameters. During free generation, the model receives the source and its own growing output; the generation function does not receive the expected target.

The pilot script creates the source/target pairs before training. The tokenizer then represents their text numerically. This is why we first separate **what answer we want to teach** from **how text becomes input to the network**. Tokenization itself does not decide which name to redact.

For now, read only Sections 1–2 of [the refresher](PREREQUISITE_KNOWLEDGE.md). Bring any unclear term back into the discussion before proceeding to embeddings.

## Training you will run

The existing checkpoints/results remain preserved. When we reach Step 7, we will prepare a separate practice output directory and explain the exact command before Nagar runs it. No new training is started merely by opening these notes. Local CPU is sufficient for the small pilot; the prepared Colab workflow remains available for a later guided session. Prior engineering tests and generated notes do not complete learning milestones.

## My Understanding

For Nagar to write after discussion. No answer has been filled in on your behalf.

## Sources

The primary sources for this route are the saved pilot manifest, `tools/pilot.py`, `tokenizer.py`, `model.py`, `data.py`, `train.py`, `generate.py` and `app.py` under `projects/Week 6/tiny-mitra`. Teaching references and worked mathematics remain in the consolidated refresher. The reference map selects *Hands-On Large Language Models* for tokenization/architecture intuition and *Deep Learning* for shapes and learning mathematics. Configuration evidence is the saved `artifacts/checkpoints/pilot_overfit/manifest.json`.
