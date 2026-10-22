# Tiny Mitra

An educational decoder-only GPT for synthetic note transformation, assembled from PyTorch primitives. Expected example:

```text
Source:   PATIENT John Lee TREATMENT DrugA
Expected: PATIENT [NAME] TREATMENT DrugA
```

The expected target is a label. Actual generation is computed by the selected checkpoint and can be wrong. No regex or reference-answer lookup produces the model's continuation.

## Joining from GitHub

Read [TEAM_START_HERE.md](TEAM_START_HERE.md) first. This source checkout contains code, notebooks, synthetic datasets, saved measurements and presentation files. The existing `.gitignore` excludes trained `.pt` checkpoints, virtual environments and demo ZIPs. A fresh clone therefore does not contain the weights needed to reproduce the saved live-demo outputs. The verified local demo ZIP includes weights and must be supplied separately; alternatively, run your own explicitly separate practice training session.

## Current evidence

The canonical submission is `submission_experiments.py` and `artifacts/mitra_final/`: all eight primary architectures and eight additional seed runs completed 4,000 updates from random weights on one frozen corpus of 1,000 synthetic examples (700 training, 100 validation, four final partitions of 50). The primary comparison uses seed 42 and `latest.pt` at update 4,000. Minimum-loss weights are retained separately. Earlier models do not initialize these weights.

The full model gets 10/11 authored notes at seed 42 and 11/11 at seeds 7 and 123. All three pass two extra clinical checks whose exact sources occur in training. The declared development-only rule selects **`final_seed7/transformer_2` for the live demo**, while primary 42 and its PHONE failure remain visible. The selected demo gets 49/50 exact notes with known names/layouts and 43/50 with new names/known layouts. It gets only 5/50 in each new-layout partition: those five successes are no-identifier controls. The primary full model gets 47, 5, 50 and 5/50 respectively. These outcomes do not establish broad redaction reliability.

All 32 primary model/partition rows and four separate demo rows have saved raw generations and audited counts. All 32 teacher-forced test loss/next-token accuracy rows were recomputed from matching weights. The controlled DrugA/DrugB test succeeds for the one-head and full primary models; the bare four-head model fails the DrugA case. Read [measured outputs](reports/RESULTS.md), [nine final findings](reports/FINAL_FINDINGS.md), [head findings](reports/HEAD_FINDINGS.md) and raw reports for their scope. Most component ablations have one seed, and three training seeds share the same evaluation notes; no statistical significance is claimed.

Both local environments pass 34 tests, with one TypedStorage deprecation warning. The fresh isolated installation also passes dependency checks, Gradio construction and real checkpoint inference. The complete 28-route API flow passes generation, attention and intervention/restoration checks. A separate live API check received 12 successive frames for the VISITED example and matched the console generator's text, probabilities and EOS. Thirty-five checkpoints have warmed CPU latency measurements; the selected demo's median was approximately 0.058 seconds for the measured 16-token note, excluding load time. This is a local observation, not a hardware-independent speed promise. All learning-notebook code cells executed on CPU; both notebooks compile. Actual Colab/GPU execution, runtime-loss recovery and GPU-to-CPU portability remain unverified.

Earlier studies remain separate: the original character study completed eight primary/eight repeat runs and32final partitions, with weak complete-note transfer; the corrected-grammar character study and initial task-piece study each completed/audited eight2,000-update runs. The initial task-piece full model reached9/11 authored notes. Broader names/digits, character exposure and a larger update budget all change in the final study, so their improvement is not a tokenizer-only causal claim. Raw failures and the deliberately stopped optional trials remain recorded in the task ledger.

Earlier browser checks displayed real final checkpoints, attention/head evidence and the four-architecture context comparison. The latest browser recheck was blocked after a connection-refused page; the updated streaming/test-metric UI still needs a human visual check. Its API checks pass. The reviewed presentation contains 28 core slides, eight technical appendix slides and detailed notes. Timed rehearsal and every presenter's explanation remain open in [the task ledger](../../../weekly%20curriculum/06-Tiny-Mitra-Project/TASKS.md). Engineering review is author self-review, not independent sign-off.

## Read and learn

Begin with [Your first session](START_HERE.md) for a short running example and a file-by-file reading path.

Read [the prerequisite guide](../../../weekly%20curriculum/06-Tiny-Mitra-Project/PREREQUISITE_KNOWLEDGE.md), then run `learning_walkthrough.ipynb` using the shared source. The canonical file path is listed below and in START_HERE.md.

Comments explain the purpose of masks, dimensions, feature splitting, target shift, loss and head intervention. B=batch, T=positions, D=feature width, H=heads, V=vocabulary. Neither a token ID nor an attention weight is an embedding parameter.

For the final path read `task_tokenizer.py` → `broad_data.py`/`assignment_data.py` → `model.py` → `task_runtime.py` → `generate.py` → `assignment_evaluate.py` → `submission_experiments.py` → `app.py`. The original character-study files remain reproducible history.

## Local environment and tested commands

Run commands from this project directory. The project-local `.venv` was created using the existing Python 3.10 installation with `--system-site-packages` to reuse PyTorch 2.1.2+cpu without another large download. Gradio 5.50.0 and pytest 8.4.2 were installed inside this environment. Its inherited global packages contain unrelated spaCy/Transformers dependency conflicts; this project does not import those model libraries. The shared repository's broken `.venv` was not modified.

An independently created `.venv-repro` now passes dependency checks, all 34 tests, Gradio construction and actual final-checkpoint inference. See [clean installation evidence](artifacts/results/clean_environment.json). It inherits no system packages. To reproduce the tested Windows/Python3.10 CPU setup in a fresh project copy:

```powershell
py -3.10 -m venv .venv-repro
./.venv-repro/Scripts/python.exe -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
./.venv-repro/Scripts/python.exe -m pip install -r requirements-cpu-tested.txt
./.venv-repro/Scripts/python.exe -m tools.verify_clean_environment
```

`requirements.txt` expresses the broader supported dependency ranges for Colab; actual compatibility there remains unverified. Record its actual runtime versions. Gradio's transitive Hub utility dependency is UI infrastructure, not a pretrained Transformer or model backend in this project.

Executed locally:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --junitxml=artifacts/results/mechanics-tests.xml
.\.venv\Scripts\python.exe data.py
.\.venv\Scripts\python.exe -m tools.pilot
.\.venv\Scripts\python.exe tools/prepare_notebooks.py
.\.venv\Scripts\python.exe app.py --port 7860
```

The initial `python -m tools.run_work` queue has finished; do not restart it merely because other work is pending. **Inspect existing workers first:** original metadata is under `artifacts/results/*_process.json`; assignment-format metadata/logs are under `artifacts/assignment_v2/results/`. Verify command lines and saved progress because PIDs can be reused. Training resumes from completed save intervals; unsaved updates cannot be recovered.

The comparison command used by that queue is:

```powershell
.\.venv\Scripts\python.exe experiments.py --steps 1000 --limit 12
```

This command being launched does not mean all runs finished; the completed manifests and measured JSON are the evidence. Other CLI options are available through `--help`. Do not change a registered protocol or resume identity silently; review/version the experiment when settings or core source change.

The current final worker runs `python submission_experiments.py`. Before launching it, inspect `artifacts/mitra_final/results/process.json`, progress and logs; an existing compatible run should resume, not duplicate. Its source/data/protocol are frozen. Earlier instructions for `experiments.py` above reproduce the original study rather than the final submission.

## Architectures

| Name | Assignment | Structure |
| --- | --- | --- |
| embedding_only | A | Token + position embeddings → vocabulary logits |
| single_attention | B | Embeddings → one bare causal head → logits |
| multi_attention | Additional | Same bare arrangement with four heads |
| transformer_1 | Additional | One full pre-LN block plus final norm |
| transformer_2 | C | Two full pre-LN blocks plus final norm |
| no_ffn | Ablation | Two blocks, FFN branches omitted |
| no_layernorm | Ablation | Two blocks, all norms omitted |
| no_residual | Ablation | Two blocks, residual additions omitted |

The bare-head comparison shares residual/norm policy. Width stays fixed; four heads split 64 features into 16 each. Depth and removal comparisons change parameter counts. Removing FFN also removes its otherwise unused pre-normalization branch. The model uses manual LayerNorm, Q/K/V, explicit score masks and softmax/value mixing; no nn.Transformer, nn.MultiheadAttention, fused attention or pretrained model is used.

## Data and training

`artifacts/dataset/manifest.json` describes 1,000 unique synthetic sources: 700 train, 100 validation and four 50-example final-test partitions. Names have separate training/validation/test pools; novel layouts are reserved for validation or final test. MRNs/phones are globally unique. Dates can recur. No-identifier controls use a shared layout and must not be described as novel-template examples. Addresses and doctor novelty are documented separately. Source and target field offsets are saved.

The tokenizer preserves ASCII spelling and whitespace with atomic control/placeholder markers. Unsupported characters and context overflow fail explicitly. The maximum full example length is about 140 tokens in the initial dataset, within the 512-position model limit. No test-fitted vocabulary or silent truncation is used.

Training scores only output targets and EOS. Source and padding prediction targets are ignored; source embeddings can still receive gradients via attention. Validation is token-weighted across all 100 validation notes. Best-checkpoint selection uses minimum validation cross-entropy. CPU preliminary runs use width 64, FFN width 192, four heads where applicable, seed 42 and a shared update budget. The tiny overfit pilot uses a separate six-note grammar and a larger learning rate; its success is not the main benchmark.

Each architecture has its own `artifacts/checkpoints/<architecture>_seed42/` directory: `best.pt`, `latest.pt`, history, manifest and progress. Checkpoints include tokenizer/configuration/data/source identity, optimizer, sampler and RNG state. Loading rejects incompatible tokenizer/configuration, and resuming rejects changed data/protocol/core source.

The corrected grammar is defined in `assignment_data.py`; `assignment_experiments.py` registers and runs its separate local comparison. Checkpoints live under `artifacts/assignment_v2/checkpoints/`. It adds complete VISITED/ON, DOB, MRN orderings, PHONE, LIVES, DIAGNOSIS, BIOMARKER and TREATMENT forms, including NSCLC, EGFR, DrugA and Chemotherapy. Context is 256 and the fixed local budget is 2,000 updates. `assignment_evaluate.py` scores replacements/clinical values at sentence anchors; generation still uses the same learned model and autoregressive loop. Eleven authored instructor/clinical prompts are development demonstrations, not final-test samples. New names include a reserved Olivia Martinez test pool; its authored demonstration must not be advertised as an untouched final test. Version-2 final partitions remain protected pending review.

## Generation and evaluation

`generate.py` appends one selected token at a time and returns actual token trace and stop reason. Greedy and sampling are separate modes; temperature does not switch the algorithm. It never fetches a gold answer, silently removes source context or substitutes random weights for missing checkpoints.

Teacher-forced loss/accuracy and free-generation quality are separate measurements. Generated-note metrics include field-aligned typed replacements, precision/recall/F1, exact match, exact-surface identifier leakage, preserved clinical field/value associations, control over-redaction, EOS validity and warm latency. Counts and selected IDs are saved. Unavailable metrics remain null. Exact-surface matching can miss partial/reformatted identifiers and is not a real-world PHI safety guarantee.

`experiments.py` registers hypotheses before benchmark runs. Its default generation diagnostics select 12 validation examples for category coverage; that subset is not the full final test. The head investigation uses eight validation diagnostics: baseline, eight single-head zero interventions, two validation-selected pairs, one-token source changes, sampled temperatures 0.1/0.7/1.2 and qualitative OOD prompts. Pairs are exploratory. Head IDs are not semantic labels and are not interchangeable across seeds. Reports require evidence review before conclusions are accepted.

## Gradio demo

Open `http://127.0.0.1:7860` after launching `app.py`.

For a hidden local server with safe project-specific restart, run `./tools/start_demo.ps1` (or add `-Restart` after UI source changes). It checks readiness for 45 seconds; on a slow first import inspect logs and wait for the local URL message. Keep the terminal/app and computer awake during the presentation.

1. **Compare models:** choose trained architectures, enter one synthetic note and inspect actual outputs/token traces. No live precision/recall is invented without a reference answer.
2. **Inspect heads:** provide a source and a clearly labeled conditional output prefix; inspect the selected next prediction's attention row. Disable a head for a separate live generation. Gates apply only to that request.
3. **Compare components:** load saved metrics, training curves and intervention results. Component variants are separately trained checkpoints, not on/off switches in a trained full model.

Requests are serialized for the small CPU demo. Models are cached by checkpoint file identity; absent checkpoints produce an explicit message. A model selection must match its checkpoint's architecture. The demo uses local weights and reports; the intended presentation does not depend on a live cloud session.

Choose `learning_pilot` for the short first-principles DrugA/DrugB demonstration. Choose `assignment_v2/<architecture>` for the instructor sentence grammar. Unprefixed architecture names select the original flat-field study; `transformer_2_latest` explicitly selects its exploratory last update. Component results keep these studies separate. The checkpoint's actual update is displayed, including a partial run if it is still training. Dataset identity is checked before loading a benchmark selection.

Choose `task_tokens/<architecture>` for the final single-corpus comparison. It loads `artifacts/mitra_final/checkpoints/<architecture>_seed42/latest.pt` with the matching tokenizer and dataset. The current full-model demo gate is all11 authored development prompts exact with valid EOS; that gate is separate from the untouched final-partition measurements and is still pending.

## Colab

`train_colab.ipynb` contains the guided workflow. `tools/prepare_notebooks.py` creates `artifacts/bundles/tiny-mitra-cloud.zip`, limited to project files, tests and synthetic data, with a SHA-256 file manifest. No book library, unrelated workspace files or credentials are bundled. Upload that ZIP in Colab and follow the notebook. Actual sign-in/Drive authorization and cloud verification require Nagar. See [the training guide](../../../weekly%20curriculum/06-Tiny-Mitra-Project/TRAINING_ENVIRONMENT.md).

New cloud work uses `assignment_train.py` and saves under `artifacts/assignment_v2_cloud/`. The notebook first runs 100 then resumes to a total of 200 pilot updates; only after reviewing actual GPU timing does it start the declared main budget. Downloaded cloud results can be selected with `python app.py --assignment-root artifacts/assignment_v2_cloud`, preserving the local study. `tools.portability` must compare an actual CUDA reference with local CPU; a CPU-only check cannot pass that gate.

## Troubleshooting

- **Missing checkpoint:** wait for the matching run or download its complete bundle; never relabel another architecture's weights.
- **Context/character error:** read the explicit message, reduce supported input or review the representation policy; no silent fallback.
- **Resume identity mismatch:** check saved source/data/configuration hashes. Preserve prior evidence and create/review a new run; do not bypass compatibility checks.
- **Poor generation despite good token accuracy:** inspect raw continuations, EOS, clinical associations and split support; do not replace predictions with expected text.
- **Windows save lock:** bounded retries handle transient locks; persistent failures remain visible in logs and must be fixed before calling the run complete.
- **Cloud disconnect:** restore durable artifacts, verify identities and resume the saved update. Actual fresh-runtime recovery is a separate verification task.
