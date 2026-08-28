import sys, json
sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open("projects/week 5/week05_tiny_transformer.ipynb", encoding="utf-8"))
cells = nb["cells"]

# Find the architecture benchmark results cell (look for "ARCHITECTURE COMPARISON RESULTS")
for i, cell in enumerate(cells):
    if cell["cell_type"] == "code" and cell.get("outputs"):
        for out in cell["outputs"]:
            if out.get("output_type") == "stream":
                text = "".join(out.get("text", []))
                if "ARCHITECTURE COMPARISON" in text or "Benchmarking" in text or "Starting primary" in text:
                    print(f"=== Cell {i} ===")
                    print(text[:5000])
