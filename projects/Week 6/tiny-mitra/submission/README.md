# Tiny Mitra — clean submission

Submit **one ZIP: `tiny-mitra-final-submission.zip`**. The full learning repository is our working history, not the submission folder.

## What the reviewer opens

```text
tiny-mitra-submission/
  README.md                 Start here: installation, demo and requirement map
  project/                  Runnable code, final data, checkpoints, tests and evidence
  presentation/             Final v7 deck and its presenter guide
  SUBMISSION_MANIFEST.json  File hashes, inclusion reasons and source identities
```

## Required and optional material

| Material | Submission decision | Why |
| --- | --- | --- |
| Manual model, tokenizer, batching, training and generation | Include | Explains and reproduces the token-to-next-token path |
| Gradio app and tested dependency files | Include | Provides the working demonstration |
| One final synthetic corpus and its split manifest | Include | Defines the shared training and held-out conditions |
| Eight primary final checkpoints at update 4,000 | Include | Makes all architecture selections run without retraining |
| Eight final repeat checkpoints and their histories | Include | Supports the reported seed comparison and selected seed-7 demo |
| Small character teaching pilot | Include, separately labelled | Makes the first example easy to explain; it is not the benchmark |
| Final raw outputs, metrics, head evidence, curves and protocols | Include under `project/artifacts/mitra_final/` | Supports the claims without crowding the entry folder |
| Final findings, demo script and verification tool | Include | Gives the reviewer a short route through results and limitations |
| Final v7 slides and final speaker guide | Include once | Current presentation artifacts |
| Earlier character, assignment-v2, task-token and continued-training studies | Omit from this ZIP; preserve in working repository | Historical experiments are not the final comparison |
| Old decks, duplicate presentation copies, screenshots and old bundles | Omit | Current deck and actual outputs are sufficient |
| Process IDs, execution queues, logs, caches and virtual environments | Omit | Machine-specific working files |
| Colab notebook, local learning notebook and refresher | Optional learning material in repository; not needed in this ZIP | The final assignment runs locally; actual Colab execution remains unverified |
| Private/copyrighted references | Omit | Not part of the project submission |

Some supporting Python files have historical names such as `assignment_data.py`, `task_experiments.py` and `train.py`. Keep them: final code imports them, the frozen source identity includes them, or tests use them. A filename alone does not establish that a module is unnecessary.

## Build the package

From `projects/Week 6/tiny-mitra` in the working repository:

```powershell
./.venv-repro/Scripts/python.exe -m tools.package_submission
```

The builder uses an explicit inclusion list. It creates a new folder under `artifacts/submission/` and the ZIP under `artifacts/bundles/`. It rejects an existing destination so it cannot silently replace another package. The original working files and earlier full demo ZIP remain intact.

The ZIP includes trained checkpoints and therefore remains outside ordinary Git tracking. The code, this guide and packaging tools can be reviewed through GitHub. Use the ZIP for the runnable submission.

## Install and run after extraction

Open the extracted `tiny-mitra-submission` folder in PowerShell:

```powershell
cd project
py -3.10 -m venv .venv-repro
./.venv-repro/Scripts/python.exe -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
./.venv-repro/Scripts/python.exe -m pip install -r requirements-cpu-tested.txt
./.venv-repro/Scripts/python.exe -m pytest -q
./.venv-repro/Scripts/python.exe app.py --submission --port 7860
```

Open `http://127.0.0.1:7860`. The submission mode shows eight final architectures, two explicitly labelled full-model repeats, and the teaching pilot. Older study choices are hidden. Model generation does not train anything.

For a different Python/platform combination, `requirements.txt` gives broader ranges; the exact installation above was tested on Windows/Python 3.10 CPU. Do not claim another runtime has been verified merely because it installs.

## Six code stops for the explanation

1. `project/task_tokenizer.py` and `tokenizer.py`: text, atomic pieces, IDs and unseen spelling.
2. `project/model.py`: token/position embeddings, custom LayerNorm, Q/K/V, masks, heads, FFN, residuals, two blocks and vocabulary projection.
3. `project/data.py`: source/output serialization, next-token shift and output-only targets.
4. `project/task_runtime.py`: loss, autograd, gradient clipping, AdamW and checkpoints.
5. `project/generate.py`: autoregressive next-token choices and EOS.
6. `project/app.py`: same inference functions behind the model dropdown and comparison views.

## Requirement-to-file map

| Assignment requirement | Evidence |
| --- | --- |
| Build the Transformer using allowed primitives | `project/model.py`; `project/tests/test_mechanics.py` |
| Tokenization and embeddings/position | `project/tokenizer.py`, `task_tokenizer.py`, `model.py` |
| Causal multi-head attention, residuals, LayerNorm, FFN, two blocks | `project/model.py`, full `transformer_2` configuration in saved checkpoints |
| Vocabulary projection and next-token prediction | `project/model.py`, `generate.py` |
| Training and saved model | `project/task_runtime.py`, `submission_experiments.py`, included `.pt` files |
| Working front end and live token trace | `project/app.py --submission` |
| Architecture/component/head comparisons | `project/artifacts/mitra_final/results/`, `project/reports/FINAL_FINDINGS.md` |
| Explain the token path | Six code stops above; final deck and speaking guide |

## Reproduction and honest limits

The primary architecture comparison uses seed 42 at update 4,000. The live demonstration uses the separately selected `final_seed7/transformer_2`; it does not replace the primary benchmark. All 16 final training runs started from random weights on the same frozen corpus. Only fixed-final checkpoints are packaged; earlier minimum-loss weights and historical studies stay in the working archive.

The selected model passes 11/11 authored demonstration prompts. It has severe unfamiliar-layout failures: 5/50 exact notes in each new-layout partition, all five being no-identifier controls. This is an educational synthetic task, not a production redaction service. Report failures alongside successes.

The unchanged final runners can reproduce training into a separate output directory. For an individual practice run use `task_train.py --help` and choose a new `--output`; do not overwrite the supplied frozen results. Reproducing training is optional for viewing the saved demo.

To verify this extracted package, run from `project`:

```powershell
./.venv-repro/Scripts/python.exe -m tools.verify_submission
```

The verifier checks file hashes, source identity, all packaged checkpoint identities, generation and head restoration, saved tables and a temporary local Gradio API session. It also runs the packaged tests. It does not certify browser appearance, the projector, team understanding, rehearsal or actual cloud execution; those remain human/runtime gates.
