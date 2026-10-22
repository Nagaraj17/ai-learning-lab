# Tiny Mitra: team starting guide

This is an educational decoder-only GPT assembled from PyTorch primitives. We implement attention, causal masking, head splitting/merging, LayerNorm, decoder blocks and autoregressive generation ourselves. We do not use pretrained model weights, `nn.Transformer`, `nn.MultiheadAttention`, or an LLM model library.

For the hand-in, follow the [clean submission guide](submission/README.md). Submit the single final-study ZIP rather than the complete working repository. Historical experiments stay available here for reference.

## Start with these files

1. [Prerequisite refresher](../../../weekly%20curriculum/06-Tiny-Mitra-Project/PREREQUISITE_KNOWLEDGE.md): tokens, embeddings, position, linear projections, attention, masks, training and evaluation. Work through one concept at a time.
2. [First learning session](START_HERE.md): trace one synthetic note through the implementation. The learning pilot and larger benchmark are distinct experiments.
3. [model.py](model.py): the shared manual model. [train.py](train.py) contains the original character/pilot training loop; [task_runtime.py](task_runtime.py) contains the final task-token training loop.
4. [Measured results](reports/RESULTS.md) and [final findings](reports/FINAL_FINDINGS.md): read the limitations alongside the successful examples.
5. [Presentation index](../../../presentations/mitra/README.md): current v7 slides and final presenter guide. Older decks are retained as history.

## What this Git checkout contains

Code, two notebooks, tests, synthetic datasets, saved experiment outputs, figures, the task ledger, and presentation files are included. Private reference materials, virtual environments, trained checkpoints and local ZIP bundles are excluded.

**A fresh clone does not include trained weights.** Gradio can only reproduce an existing trained model's generation after that model's matching checkpoint is available. The saved measurements and recorded-output fallback can be read without running inference; the fallback performs no model computation.

Nagar already has a separately verified `artifacts/bundles/tiny-mitra-demo.zip` containing code, matching weights and reports. It must be supplied separately for the existing live demonstration. There is no checkpoint download hosted by this source branch. Keep each checkpoint with its matching configuration and tokenizer; do not substitute randomly initialized weights when a checkpoint is missing.

## Get the source

```powershell
git clone --branch master https://github.com/Nagaraj17/ai-learning-lab.git
cd 'ai-learning-lab/projects/Week 6/tiny-mitra'
```

See [README.md](README.md) for the tested Windows/Python 3.10 CPU installation. From the project directory, its isolated setup is:

```powershell
py -3.10 -m venv .venv-repro
./.venv-repro/Scripts/python.exe -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
./.venv-repro/Scripts/python.exe -m pip install -r requirements-cpu-tested.txt
./.venv-repro/Scripts/python.exe -m pytest -q
```

After obtaining and extracting the matching demo bundle, follow its README and run `./tools/start_demo.ps1` from the extracted project. Gradio's generation button performs inference; it does not train. Read `learning_walkthrough.ipynb` one cell at a time before starting your own practice training. Do not restart the completed benchmark queue merely to explore the code.

`train_colab.ipynb` is provided, but actual Colab/GPU execution and interruption recovery have not been verified. Local CPU execution has been verified. No cloud purchase is required to read or review the project.

## What we can honestly demonstrate

The four-note teaching pilot is a small learning example. The final benchmark contains eight architectures and eight additional seed runs on one frozen synthetic corpus. The development-selected demo checkpoint passes 11/11 authored notes but fails most novel-layout test notes. Read the exact partitions and denominators in the reports; this is not a production redaction system.

Passing tests and preparing slides do not establish anyone's understanding. Every presenter still needs to explain the token-to-prediction path and rehearse their section. Track those open gates in [TASKS.md](../../../weekly%20curriculum/06-Tiny-Mitra-Project/TASKS.md).
