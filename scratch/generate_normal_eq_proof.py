import matplotlib.pyplot as plt
import os

# Create figure
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
ax.axis('off')

# Title
ax.text(0.5, 0.95, "Derivation of the Normal Equation", 
        fontsize=20, fontweight='bold', ha='center', va='top', color='#1e293b')
ax.text(0.5, 0.90, "Why we Left-Multiply Xᵀ on both sides", 
        fontsize=14, fontstyle='italic', ha='center', va='top', color='#64748b')

# Box 1: Starting Equation
box1 = dict(boxstyle='round,pad=0.8', facecolor='#f1f5f9', edgecolor='#cbd5e1', linewidth=2)
ax.text(0.5, 0.78, r"Step 1: Starting Equation" + "\n" + r"$\mathbf{X} \mathbf{w} = \mathbf{y}$" + "\n" + r"Shapes: $(1000 \times 2) \cdot (2 \times 1) = (1000 \times 1)$",
        fontsize=14, ha='center', va='center', bbox=box1, color='#0f172a')

# Arrow 1
ax.annotate('', xy=(0.5, 0.65), xytext=(0.5, 0.70),
            arrowprops=dict(arrowstyle="->", color='#2563eb', lw=3))

# Box 2: Left Multiply X^T
box2 = dict(boxstyle='round,pad=0.8', facecolor='#eff6ff', edgecolor='#93c5fd', linewidth=2)
ax.text(0.5, 0.57, r"Step 2: Multiply $\mathbf{X}^\top$ from the LEFT on both sides" + "\n" + r"$\mathbf{X}^\top (\mathbf{X} \mathbf{w}) = \mathbf{X}^\top \mathbf{y}$" + "\n" + r"$(\mathbf{X}^\top \mathbf{X}) \mathbf{w} = \mathbf{X}^\top \mathbf{y}$" + "\n" + r"Shapes: $[(2 \times 1000) \cdot (1000 \times 2)] \cdot (2 \times 1) = (2 \times 1000) \cdot (1000 \times 1)$" + "\n" + r"Resulting Shapes: $(2 \times 2) \cdot (2 \times 1) = (2 \times 1)$  ✓ Match!",
        fontsize=13, ha='center', va='center', bbox=box2, color='#1e3a8a')

# Arrow 2
ax.annotate('', xy=(0.5, 0.43), xytext=(0.5, 0.48),
            arrowprops=dict(arrowstyle="->", color='#2563eb', lw=3))

# Box 3: Left Multiply Inverse
box3 = dict(boxstyle='round,pad=0.8', facecolor='#f0fdf4', edgecolor='#86efac', linewidth=2)
ax.text(0.5, 0.33, r"Step 3: Multiply $(\mathbf{X}^\top \mathbf{X})^{-1}$ from the LEFT on both sides" + "\n" + r"$(\mathbf{X}^\top \mathbf{X})^{-1} (\mathbf{X}^\top \mathbf{X}) \mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$" + "\n" + r"Since $\mathbf{A}^{-1} \mathbf{A} = \mathbf{I}$ (Identity Matrix):" + "\n" + r"$\mathbf{I} \cdot \mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$",
        fontsize=13, ha='center', va='center', bbox=box3, color='#14532d')

# Arrow 3
ax.annotate('', xy=(0.5, 0.20), xytext=(0.5, 0.25),
            arrowprops=dict(arrowstyle="->", color='#16a34a', lw=3))

# Box 4: Final Result
box4 = dict(boxstyle='round,pad=0.8', facecolor='#dcfce7', edgecolor='#22c55e', linewidth=3)
ax.text(0.5, 0.12, r"Final Normal Equation Result:" + "\n" + r"$\mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$" + "\n" + r"Shape: $(2 \times 2) \cdot (2 \times 1) = \mathbf{(2 \times 1)}$  ✓",
        fontsize=15, fontweight='bold', ha='center', va='center', bbox=box4, color='#15803d')

# Save figure
os.makedirs('topics/images', exist_ok=True)
output_path = 'topics/images/normal_equation_proof.png'
plt.tight_layout()
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Saved image to {output_path}")
