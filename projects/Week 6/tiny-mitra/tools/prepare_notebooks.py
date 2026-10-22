"""Generate thin notebooks and a project-only cloud bundle, never the book library."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def notebook(cells):
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}, "cells": [{"id": f"cell-{index}", "cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True), **({"outputs": [], "execution_count": None} if kind == "code" else {})} for index, (kind, text) in enumerate(cells)]}


def main():
    # These are the actual current cells, without historical override layers.
    learning = [('markdown',
      "# Tiny Mitra: one token's journey\n"
      'Run this notebook from the project directory. It imports the same Python files as training and '
      'Gradio. Read the prerequisite guide alongside it. Random weights illustrate calculations; they '
      'are not learned results.'),
     ('code',
      'from pathlib import Path\n'
      'import torch\n'
      'from tokenizer import CharacterTokenizer\n'
      'from config import ModelConfig\n'
      'from model import TinyGPT\n'
      'from data import render, batch_items\n'
      'torch.set_num_threads(2)\n'
      'torch.manual_seed(42)\n'
      'tokenizer = CharacterTokenizer()\n'
      "source = 'PATIENT John Lee TREATMENT DrugA'\n"
      "record = render([('PATIENT','John Lee'),('TREATMENT','DrugA')])\n"
      'record'),
     ('markdown',
      '## Text → tokens → IDs\n'
      'The table address is an integer ID; it is not an embedding. Special markers and placeholders '
      'are atomic. New ASCII names retain their spelling.'),
     ('code',
      'ids, offsets = tokenizer.encode_with_offsets(source)\n'
      'list(zip(ids, [tokenizer.tokens[i] for i in ids], offsets))'),
     ('markdown',
      '## Token identity + position\n'
      'An embedding table has one learned row per token. Position has a separate row per index. '
      'Addition preserves the feature width. These initial random vectors have no established clinical '
      'interpretation.'),
     ('code',
      'model = TinyGPT(ModelConfig(len(tokenizer), d_model=16, d_ff=32, context=256))\n'
      'x = torch.tensor([ids])\n'
      'token_vectors = model.token_embedding(x)\n'
      'position_vectors = model.position_embedding(torch.arange(x.shape[1]))\n'
      'combined = token_vectors + position_vectors\n'
      "print('IDs:', x.shape, 'token vectors:', token_vectors.shape, 'positions:', "
      "position_vectors.shape, 'combined:', combined.shape)\n"
      "print('First token:', token_vectors[0,0].detach(), '\\nPosition:', "
      "position_vectors[0].detach(), '\\nSum:', combined[0,0].detach())"),
     ('markdown',
      '## Inside one head: Q, K, V and the causal mask\n'
      'Q and K compare token features; V supplies the values to mix. B is batch size, T is token '
      'count, D is feature width, H is the number of heads and Dh=D/H. Scores have shape B,H,T,T. Each '
      'query row distributes probability over its present/past keys. These are random teaching '
      'weights, so no semantic role is inferred.'),
     ('code',
      'import math\n'
      'with torch.no_grad():\n'
      '    block = model.blocks[0]\n'
      '    attention = block.attention\n'
      '    normalized = block.norm1(combined)\n'
      '    B,T,D = normalized.shape\n'
      '    H,Dh = attention.heads,attention.head_width\n'
      '    def split(projection):\n'
      '        return projection(normalized).view(B,T,H,Dh).transpose(1,2)\n'
      '    Q,K,V = split(attention.q),split(attention.k),split(attention.v)\n'
      '    scores = Q @ K.transpose(-2,-1) / math.sqrt(Dh)\n'
      '    future = torch.ones(T,T,dtype=torch.bool).triu(1)\n'
      "    weights = torch.softmax(scores.masked_fill(future,float('-inf')),dim=-1)\n"
      '    mixed = weights @ V\n'
      '    merged = mixed.transpose(1,2).contiguous().view(B,T,D)\n'
      '    attention_update = attention.output(merged)\n'
      '    assert torch.allclose(attention_update,attention(normalized))\n'
      '    assert torch.allclose(weights.sum(-1),torch.ones(B,H,T))\n'
      '    assert torch.count_nonzero(weights.masked_select(future[None,None])) == 0\n'
      "    print('Q/K/V:',Q.shape,'scores:',scores.shape,'mixed:',mixed.shape)\n"
      "    print('First query, head 0:',weights[0,0,0])\n"
      "    print('Residual keeps width:',(combined+attention_update).shape)"),
     ('markdown',
      '## Finish a decoder block\n'
      'The first residual adds the attention update to the input. A second normalization and FFN '
      'produce another update, followed by another residual addition. The next block repeats this '
      'process. Final normalization and the vocabulary projection produce one raw score per possible '
      'next token. Softmax converts scores to probabilities; choosing a token is a separate generation '
      'decision.'),
     ('code',
      'with torch.no_grad():\n'
      '    after_attention = combined + attention_update\n'
      '    after_ffn = after_attention + block.ffn(block.norm2(after_attention))\n'
      '    assert torch.allclose(after_ffn,block(combined,None))\n'
      "    print('Block output:',after_ffn.shape)\n"
      '    final_logits = model(x)\n'
      '    next_probabilities = torch.softmax(final_logits[0,-1],dim=-1)\n'
      "    print('Vocabulary scores:',final_logits.shape,'probability "
      "sum:',float(next_probabilities.sum()))\n"
      "    print('Random-model candidate:',tokenizer.tokens[int(next_probabilities.argmax())])"),
     ('markdown',
      '## Training alignment\n'
      'At `<OUTPUT>`, the next target is `P`. Before that, targets have ignore value -100. EOS remains '
      'scored. Padding is a separate mask.'),
     ('code',
      'inputs, targets, valid = batch_items([record], tokenizer, model.config.context)\n'
      '[(tokenizer.tokens[int(i)], None if int(y)==-100 else tokenizer.tokens[int(y)]) for i,y in '
      'zip(inputs[0],targets[0])]'),
     ('markdown',
      '## Forward → loss → gradient\n'
      'Loss uses teacher forcing, so expected previous output tokens are present. This single backward '
      'pass calculates gradients; no optimizer update is made here.'),
     ('code',
      'logits = model(inputs, valid, capture=True)\n'
      'loss = torch.nn.functional.cross_entropy(logits.reshape(-1,len(tokenizer)), '
      'targets.reshape(-1), ignore_index=-100)\n'
      'model.zero_grad()\n'
      'loss.backward()\n'
      "print('Logits:', logits.shape, 'loss:', float(loss), 'embedding gradient norm:', "
      'float(model.token_embedding.weight.grad.norm()))\n'
      'attention = model.attention_matrices()[0]\n'
      "print('Block 0 attention B,H,T,T:', attention.shape)\n"
      "print('Final query, head 0:', attention[0,0,-1])"),
     ('markdown',
      '## Real checkpoint and free generation\n'
      'This section uses trained weights only if present. A generated output can differ from the '
      'expected target. Inspect the actual result before interpreting it.'),
     ('code',
      'from train import load_checkpoint\n'
      'from generate import generate\n'
      "checkpoint_path = Path('artifacts/checkpoints/transformer_2_seed42/best.pt')\n"
      'if checkpoint_path.exists():\n'
      '    trained, saved_tokenizer, metadata = load_checkpoint(checkpoint_path)\n'
      "    print(metadata['run_id'], 'selected step', metadata['step'])\n"
      '    print(generate(trained, saved_tokenizer, source))\n'
      'else:\n'
      "    print('No full-model checkpoint yet. Train or download one; do not substitute random "
      "weights.')"),
     ('markdown',
      '## Optional: the fixed-task-piece representation experiment\n'
      'The first walkthrough used the character teaching model. The new experiment encodes declared '
      'schema/clinical pieces as single IDs and retains character spelling for other text. These are '
      'fixed pieces, not a vocabulary fitted to test names; a piece may also occur inside a longer '
      'word. The decoder, attention and gradient mathematics remain the same. Compare complete-note '
      'metrics across representations; per-token loss has different units.'),
     ('code',
      'from task_tokenizer import TaskTokenizer\n'
      'from task_runtime import load_task_checkpoint\n'
      'task_tokenizer=TaskTokenizer()\n'
      'for representation in (tokenizer,task_tokenizer):\n'
      '    encoded=representation.encode(source)\n'
      "    print(representation.metadata()['type'],len(encoded),[representation.tokens[i] for i in "
      'encoded])\n'
      "task_checkpoint=Path('artifacts/mitra_final/checkpoints/transformer_2_seed42/latest.pt')\n"
      'if task_checkpoint.exists():\n'
      '    task_model,task_tokenizer,task_metadata=load_task_checkpoint(task_checkpoint)\n'
      "    print('Actual saved update:',task_metadata['step'])\n"
      '    print(generate(task_model,task_tokenizer,source))\n'
      'else:\n'
      "    print('Task-piece checkpoint not available yet; no result is invented.')"),
     ('markdown',
      '## My understanding\n'
      'Write in your own words after reading and discussing: what does an ID select, what does '
      'position add, which targets contribute loss, and how does free generation differ from teacher '
      'forcing? We will discuss these one at a time. This notebook does not automatically mark '
      'learning complete.')]
    cloud = [('markdown',
      '# Tiny Mitra — guided Colab training\n'
      'Prepared locally; actual Colab execution is pending. This notebook trains our manual PyTorch '
      'models from random weights on the final saved corpus of1,000 synthetic examples, using fixed '
      'task pieces and character fallback. Review the actual GPU pilot before the4,000-update main '
      'budget. Earlier character/continued-training experiments are separate history, not '
      'initialization for the final models.'),
     ('markdown',
      '## 1. Upload the project bundle\n'
      'In Colab use Runtime → Change runtime type and choose an available GPU. The next cell opens a '
      'file picker: choose `tiny-mitra-cloud.zip` produced by tools/prepare_notebooks.py. Only project '
      'source, tests and synthetic data are bundled.'),
     ('code',
      'from google.colab import files\n'
      'from pathlib import Path\n'
      'import os, zipfile, hashlib, json, sys, subprocess\n'
      'uploaded = files.upload()\n'
      "archive = next(name for name in uploaded if name.endswith('.zip'))\n"
      "root = Path('/content/tiny-mitra').resolve()\n"
      'root.mkdir(exist_ok=True)\n'
      'with zipfile.ZipFile(archive) as bundle:\n'
      '    for member in bundle.namelist():\n'
      "        assert (root / member).resolve().is_relative_to(root), 'Archive path escapes project'\n"
      '    bundle.extractall(root)\n'
      'os.chdir(root)\n'
      "manifest = json.loads(Path('bundle_manifest.json').read_text())\n"
      'for name, expected in manifest.items():\n'
      '    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name\n'
      "print('Project bundle verified:', len(manifest), 'files')"),
     ('markdown',
      '## 2. Dependencies and actual GPU check\n'
      'This verifies allocated hardware rather than assuming that selecting GPU succeeded. Package '
      'installation may take a few minutes.'),
     ('code',
      "subprocess.run([sys.executable,'-m','pip','install','-r','requirements.txt'],check=True)\n"
      'import torch\n'
      "print('Python:',sys.version,'PyTorch:',torch.__version__)\n"
      "assert torch.cuda.is_available(), 'No GPU allocated. Reconnect to an available GPU or discuss "
      "the CPU/alternative plan.'\n"
      "print('GPU:',torch.cuda.get_device_name(0))\n"
      "device='cuda'"),
     ('markdown',
      '## 3. Correctness and data\n'
      'Passing these checks is necessary before training. The GPU smoke check explicitly places model '
      'and IDs on CUDA.'),
     ('code',
      "subprocess.run([sys.executable,'-m','pytest','-q'],check=True)\n"
      "DATASET_PATH='artifacts/mitra_final/dataset/examples.json'\n"
      "assert Path(DATASET_PATH).exists(),'Use the current project bundle containing the synthetic "
      "corpus'\n"
      'from broad_data import validate_broad\n'
      'print(validate_broad(json.loads(Path(DATASET_PATH).read_text())))\n'
      'from model import TinyGPT\n'
      'from config import ModelConfig\n'
      'from task_tokenizer import TaskTokenizer\n'
      'tokenizer=TaskTokenizer()\n'
      'smoke=TinyGPT(ModelConfig(len(tokenizer),context=256)).to(device)\n'
      "print('GPU logits shape:',smoke(torch.tensor([[2,4,12]],device=device)).shape)\n"
      'del smoke\n'
      "subprocess.run([sys.executable,'-m','tools.pilot','--device',device,'--output','artifacts/checkpoints/cloud_tiny_pilot','--report','artifacts/results/cloud_tiny_pilot.json'],check=True)"),
     ('markdown',
      '## 4. Authorize Drive and restore saved work\n'
      'The permission dialog is yours to approve. Runtime disk is temporary; Drive stores completed '
      'checkpoint copies. The notebook only uses `MyDrive/TinyMitra` for this project.'),
     ('code',
      'from google.colab import drive\n'
      'import shutil,time\n'
      "drive.mount('/content/drive')\n"
      "persistent=Path('/content/drive/MyDrive/TinyMitra')\n"
      'persistent.mkdir(exist_ok=True)\n'
      "if (persistent/'artifacts').exists():\n"
      "    shutil.copytree(persistent/'artifacts',Path('artifacts'),dirs_exist_ok=True)\n"
      "print('Restored available artifacts; run identity is checked by train.py')"),
     ('markdown',
      '## 5. Timed GPU pilot and interruption/resume\n'
      'Run 100 updates first. The second command resumes to a TOTAL of 200 updates, not another 200. '
      'Review the recorded validation curves and elapsed time before choosing the main comparison '
      'budget. This is a pilot, not a final result.'),
     ('code',
      "pilot=['--architecture','transformer_2','--device',device,'--dataset',DATASET_PATH,'--output','artifacts/task_cloud_pilot','--validate-every','50']\n"
      "subprocess.run([sys.executable,'task_train.py',*pilot,'--steps','100','--resume'],check=True)\n"
      "subprocess.run([sys.executable,'task_train.py',*pilot,'--steps','200','--resume'],check=True)\n"
      "print(Path('artifacts/task_cloud_pilot/history.json').read_text())"),
     ('markdown',
      '## 6. Main runs with durable checkpoint copies\n'
      'Set STEPS only after the pilot review. An interrupted run resumes using the same '
      'configuration/source/data. Keep the runtime open for interactive work; ordinary Colab '
      'availability/session limits still apply.'),
     ('code',
      'def persist_artifacts():\n'
      "    for source in Path('artifacts').rglob('*'):\n"
      "        if source.is_file() and source.suffix != '.tmp':\n"
      '            target=persistent/source\n'
      '            target.parent.mkdir(parents=True,exist_ok=True)\n'
      "            temporary=target.with_name(target.name+'.copying')\n"
      '            shutil.copy2(source,temporary)\n'
      '            temporary.replace(target)\n'
      'STEPS=4000  # Freeze after real GPU pilot review; this is a separate broad-only study.\n'
      'from config import ARCHITECTURES\n'
      'from task_runtime import task_source_hash\n'
      'from data import digest\n'
      'cloud_records=json.loads(Path(DATASET_PATH).read_text())\n'
      "cloud_protocol={'steps':STEPS,'architectures':list(ARCHITECTURES),'seed':42,'dataset_hash':digest(cloud_records),'source_hash':task_source_hash(),'device':device,'gpu':torch.cuda.get_device_name(0),'selection':'fixed "
      "final update; best loss saved separately','initialization':'random weights on the same frozen "
      "final corpus; local and cloud artifacts stay separate'}\n"
      "protocol_path=Path('artifacts/task_tokens_cloud/results/protocol.json')\n"
      'protocol_path.parent.mkdir(parents=True,exist_ok=True)\n'
      'if protocol_path.exists():\n'
      "    assert json.loads(protocol_path.read_text())==cloud_protocol,'Changed protocol: use a "
      "separately reviewed directory'\n"
      'else:\n'
      '    protocol_path.write_text(json.dumps(cloud_protocol,indent=2))\n'
      'for architecture in ARCHITECTURES:\n'
      '    '
      "command=[sys.executable,'task_train.py','--architecture',architecture,'--device',device,'--dataset',DATASET_PATH,'--steps',str(STEPS),'--output',f'artifacts/task_tokens_cloud/checkpoints/{architecture}_seed42','--resume']\n"
      '    process=subprocess.Popen(command)\n'
      '    while process.poll() is None:\n'
      '        time.sleep(30)\n'
      '        persist_artifacts()\n'
      '    persist_artifacts()\n'
      "    assert process.returncode==0,'Run failed; inspect output and saved state'"),
     ('markdown',
      '## 7. Development evaluation and export\n'
      'The cloud main run starts from random weights on the same final synthetic corpus and '
      'declared4,000-update budget. It is unverified until actually executed; hardware/software '
      'differences and recovery still need checks. Authored/validation diagnostics are not final-test '
      'performance. After download/extraction, use `python app.py --task-root '
      'artifacts/task_tokens_cloud` while preserving local results.'),
     ('code',
      "subprocess.run([sys.executable,'-m','tools.evaluate_task','--device',device,'--dataset',DATASET_PATH],check=True)\n"
      "subprocess.run([sys.executable,'-m','tools.portability','--write-reference','--device',device,'--tokenizer-kind','task','--checkpoint','artifacts/task_tokens_cloud/checkpoints/transformer_2_seed42/latest.pt','--reference','artifacts/task_tokens_cloud/results/portability_reference.json'],check=True)\n"
      'persist_artifacts()\n'
      "archive=shutil.make_archive('/content/tiny-mitra-trained','zip','artifacts')\n"
      'files.download(archive)'),
     ('markdown',
      '## Remaining acceptance evidence\n'
      'Record real GPU/version/timing. After a saved checkpoint, disconnect and restore in a fresh '
      'runtime; compare with an uninterrupted control. Two commands in one runtime alone do not prove '
      'runtime-loss recovery. On your laptop run `python -m tools.portability --tokenizer-kind task '
      '--checkpoint artifacts/task_tokens_cloud/checkpoints/transformer_2_seed42/latest.pt --reference '
      'artifacts/task_tokens_cloud/results/portability_reference.json`. Check actual '
      'gpu_to_cpu_passed, logit tolerance and greedy continuation. Learning, cloud verification and '
      'final-test release remain separate gates.')]
    for name, cells in (("learning_walkthrough.ipynb", learning), ("train_colab.ipynb", cloud)):
        value = notebook(cells)
        for index, cell in enumerate(value["cells"]):
            if cell["cell_type"] == "code": compile("".join(cell["source"]), f"{name}:{index}", "exec")
        (ROOT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    paths = list(ROOT.glob("*.py")) + list((ROOT / "tests").glob("*.py")) + list((ROOT / "tools").glob("*.py"))
    paths += list((ROOT / "tools").glob("*.ps1"))
    paths += [ROOT / name for name in ("requirements.txt", "requirements-cpu-tested.txt", "README.md", "START_HERE.md", "learning_walkthrough.ipynb", "train_colab.ipynb") if (ROOT / name).exists()]
    paths += list((ROOT / "artifacts/dataset").glob("*.json"))
    paths += list((ROOT / "artifacts/assignment_v2/dataset").glob("*.json"))
    paths += list((ROOT / "artifacts/mitra_final/dataset").glob("*.json"))
    manifest = {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    target = ROOT / "artifacts/bundles/tiny-mitra-cloud.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in paths: bundle.write(path, str(path.relative_to(ROOT)))
        bundle.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
    print(f"Two notebook cell sets compile; cloud bundle: {len(manifest)} project files, {target.stat().st_size} bytes")


if __name__ == "__main__":
    main()
