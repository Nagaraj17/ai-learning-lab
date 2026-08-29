import json

nb_path = 'projects/week 5/week05_tiny_transformer.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_disclaimer = """# Week 5 — Tiny Causal Transformer from Scratch in Pure NumPy
## A Generalization Experiment on Synthetic PA Step-Therapy Workflows

> **Research Question**: Does progressively adding contextual attention, multiple attention heads,
> FFN processing, residual connections, LayerNorm and Transformer depth improve next-event
> prediction on *unseen* prior-authorization step-therapy workflows?

### ⚠️ A Quick Note on the Healthcare Data
If you are new to healthcare, think of **Step Therapy** like trying to get your company's IT department to buy you a new $2,000 laptop. They will first make you restart it (the cheap, standard "step"). If that fails, they might reinstall Windows. Only if that fails will they approve the expensive new laptop. Insurance companies do the exact same thing with expensive medications—you must try the cheaper generic drugs first. 

When a doctor formally asks the insurance company for permission to skip to the expensive drug, that request is called a **Prior Authorization (PA)**.

In the real world, AI is used to act as an ultra-smart administrative assistant—predicting denials so doctors can attach missing documents *before* they click submit, saving patients weeks of waiting.

To study this safely in our lab, the data in this notebook is **completely fictional**:
* We use a simulated "Sandbox" generator that mimics the back-and-forth between doctors and insurance companies. 
* The workflow events are *inspired by* the real-world HL7 Da Vinci standard, but the medications (like ZynPhase-X) and the approval rules are completely made up for learning purposes. 
* This AI is learning to reverse-engineer our simulated rules, not making real medical decisions!
"""

# Format as a list of strings with newlines like jupyter expects
source_lines = [line + '\n' for line in new_disclaimer.split('\n')]
# Remove the last trailing newline for exactly matching standard markdown block ending
source_lines[-1] = source_lines[-1][:-1]

# Update the first cell
nb['cells'][0]['source'] = source_lines

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    
print("Notebook disclaimer updated successfully!")
