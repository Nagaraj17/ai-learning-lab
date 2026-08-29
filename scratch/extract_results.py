import sys, json
sys.stdout.reconfigure(encoding='utf-8')
nb = json.load(open("projects/week 5/week05_tiny_transformer.ipynb", encoding="utf-8"))
cells = nb["cells"]

print(f"Total cells: {len(cells)}")
print(f"Cells with outputs: {sum(1 for c in cells if c.get('outputs'))}")
print()

# Print all text outputs from code cells
for i, cell in enumerate(cells):
    if cell["cell_type"] == "code" and cell.get("outputs"):
        print(f"{'='*60}")
        print(f"Cell {i} outputs:")
        for out in cell["outputs"]:
            if out.get("output_type") == "stream":
                text = "".join(out.get("text", []))
                if text.strip():
                    print(text[:3000])  # cap at 3000 chars per cell
            elif out.get("output_type") == "error":
                print(f"ERROR: {out.get('ename')}: {out.get('evalue')}")
