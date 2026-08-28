import sys, json
sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open("projects/week 5/week05_tiny_transformer.ipynb", encoding="utf-8"))
cells = nb["cells"]

# Find save path mentions in outputs
for i, cell in enumerate(cells):
    if cell["cell_type"] == "code" and cell.get("outputs"):
        for out in cell["outputs"]:
            if out.get("output_type") == "stream":
                text = "".join(out.get("text", []))
                if "Saved:" in text or "saved" in text.lower():
                    print(f"Cell {i}: {text.strip()[:200]}")
