import json
nb = json.load(open("projects/week 5/week05_tiny_transformer.ipynb"))
cells = nb["cells"]
print(f"Total cells: {len(cells)}")
for i, c in enumerate(cells):
    src = "".join(c["source"])
    preview = src[:80].replace("\n", "\\n")
    print(f"Cell {i:2d}: {c['cell_type']:8s} | chars={len(src):5d} | {preview}")
