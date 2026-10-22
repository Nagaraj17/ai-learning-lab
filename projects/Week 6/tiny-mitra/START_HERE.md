# Your first session with Tiny Mitra

Start with **one short note and one model**. The project has several experiments because we need comparison evidence; you can learn the common path first.

## 1. See a real small model run — about five minutes

From this project directory in PowerShell:

```powershell
./tools/start_demo.ps1
```

Open `http://127.0.0.1:7860`. On **Compare models**, keep `learning_pilot` selected and click **Generate and compare**. The saved teaching checkpoint was observed to generate:

```text
Input:   PATIENT Jo TREATMENT DrugA
Output:  PATIENT [NAME] TREATMENT DrugA
```

Change `DrugA` to `DrugB` and generate again. Notice that the name is replaced while the treatment follows the input. Expand **Actual generated token trace** to see the model's individual choices. This pilot learned four short training notes; its success does not establish general redaction quality.

Then select `final_seed7/transformer_2` and try `PATIENT John Smith VISITED Dr Jones ON January 12`. This is the full two-block development demo model, trained from random weights on the final synthetic corpus. Its11/11 authored checks are separate from the primary seed42architecture comparison and from final-test performance. We will explore those distinctions after the first token walkthrough.

Expand **Watch actual next-token predictions**, select that same checkpoint and click **Generate tokens one at a time**. Each update follows an actual model prediction. The growing trace shows the chosen token's ID and probability; EOS ends generation. This API path is checked against `generate.py`. The latest visual browser recheck remains a human gate.

## 2. Understand what entered the model — about ten minutes

Read the opening sections of the [refresher](../../../weekly%20curriculum/06-Tiny-Mitra-Project/PREREQUISITE_KNOWLEDGE.md), then open `learning_walkthrough.ipynb` from the project directory. Run one cell at a time while we discuss it.

For our implementation:

```text
Text:       Jo
Tokens:     J, o                 two character tokens
IDs:        two integer addresses in our fixed vocabulary
Embeddings: one learned numerical vector per selected address
Position:   another learned vector for each sequence position
Model input at a position = token vector + position vector
```

The tokenizer splits/encodes text. It does not learn the embedding vectors. The teaching pilot uses characters plus atomic markers. The final comparison also gives fixed task pieces such as `PATIENT` and `DrugA` their own IDs, while other spelling uses characters. A fixed piece can occur inside a longer word; decoding preserves the text. New supported names remain representable. A representable character still needs useful learned behavior; the final data checks letter exposure in training names. The notebook prints actual IDs and vectors.

## 3. Follow the code in small pieces

| Your question | Read | What to look for |
| --- | --- | --- |
| How does text become IDs? | `task_tokenizer.py`, `TaskTokenizer`; base mechanics in `tokenizer.py` | Fixed task pieces/characters, atomic markers, encode/decode and offsets |
| What answer are we teaching? | `assignment_data.py`, `instructor_examples`; final values in `broad_data.py` | A source sentence, expected target, spans and representative values; these are supervision/evaluation labels |
| Which next tokens count during training? | `data.py`, `training_item` | Concatenated source/output protocol, one-position shift, ignored source targets and scored EOS |
| Where do embeddings and position enter? | `model.py`, `TinyGPT.forward` | Two lookups and their sum |
| How can the current token use earlier context? | `model.py`, `CausalAttention.forward` | Q/K/V, scaled scores, masks, softmax, value mixture, head merge |
| What is a complete block? | `model.py`, `DecoderBlock.forward` | Pre-normalization, attention, residual, FFN and residual |
| How are weights learned and saved? | `task_runtime.py`, `train_task` | Forward, loss, zero gradients, backward, gradient check/clip, AdamW update, validation and checkpoint |
| How is text produced afterward? | `generate.py`, `generate` | Load learned weights; predict one token, append and repeat until EOS/limit |
| How do we judge results? | `evaluate.py`, `assignment_evaluate.py` | Raw generated text, exact notes, correct replacement locations, clinical preservation and denominators |
| How does Gradio use all of this? | `app.py`, `selected_model` and `compare` | Selection loads matching weights and calls the same generation function |

`task_train.py` starts **one training run** using the shared `task_runtime.py` loop. `submission_experiments.py` runs the eight final architectures under the same registered settings. Gradio loads checkpoints for inference; its generation button does not train. Older character/representation drivers remain for reproducing earlier experiments. Start with the final route above.

## 4. Read the correct experiment family

| UI choice | What it means |
| --- | --- |
| `learning_pilot` | Small four-note teaching model; first demonstration |
| `task_tokens/transformer_2` and other `task_tokens/` names | Completed final comparison: one corpus of 1,000 examples, random initial weights, 4,000 updates per architecture; reviewed outcomes include severe new-layout failures |
| `final_seed7/transformer_2`, `final_seed123/transformer_2` | Explicitly identified full-model repeats, available after4,000updates; seed changes both initialization and minibatch order. Primary benchmark remains seed42 |
| `assignment_v2/transformer_2` and other `assignment_v2/` names | Earlier corrected-grammar character study |
| Unprefixed names such as `transformer_2` | Original flat-field study, with its own recorded limitations |
| `transformer_2_latest` | Last update of the original full model; exploratory comparison with its best-loss checkpoint |

Every architecture has its own trained weights. All models share the tokenizer/model/training/generation machinery where appropriate. A separately trained no-residual model and a temporary disabled head answer different questions.

Current run completion and quality live in [the task ledger](../../../weekly%20curriculum/06-Tiny-Mitra-Project/TASKS.md) and [measured report](reports/RESULTS.md). The [offline fallback](reports/demo_fallback.html) displays saved actual outputs if the live demo is unavailable; it performs no inference.

We will explain and demonstrate each concept before asking you to explain it back, one question at a time. Learning milestones remain open until you can trace the prompt yourself. Team speaking roles, actual Colab sign-in/GPU checks and timed rehearsal remain separate human steps.
