"""
Generate static architecture diagrams for the Week 5 notebook.
Visual A: PAS-inspired workflow diagram
Visual B: Architecture ladder (Models A → D)
Visual C: Pre-LN Transformer block diagram
"""
import sys, os
sys.path.insert(0, "projects/week 5")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

os.makedirs("projects/week 5/visualizations", exist_ok=True)


# ============================================================
# VISUAL A: PAS-Inspired Workflow Diagram
# ============================================================
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#f8f9fa')

ax.set_title("PAS-Inspired Prior Authorization Workflow\n(Educational/Fictional — Not Real Clinical Guidance)",
             fontsize=13, fontweight='bold', pad=15)

def box(ax, x, y, w, h, label, color='#3498db', text_color='white', fontsize=9, actor=None):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.1", facecolor=color,
                           edgecolor='white', linewidth=1.5, zorder=5)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', zorder=6, wrap=True,
            multialignment='center')
    if actor:
        ax.text(x, y - h/2 - 0.18, f'[{actor}]', ha='center', va='top',
                fontsize=7, color='gray', zorder=6)

def arrow(ax, x1, y1, x2, y2, label='', color='#555555'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5, connectionstyle='arc3,rad=0'))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.12, my, label, fontsize=7.5, color='#555', zorder=7)

# Provider swim lane
ax.axhspan(6.5, 12, alpha=0.06, color='#3498db')
ax.text(0.3, 9.25, 'PROVIDER', fontsize=9, color='#2980b9', fontweight='bold',
        rotation=90, va='center', ha='center')

# Payer swim lane
ax.axhspan(0, 6.5, alpha=0.06, color='#e74c3c')
ax.text(0.3, 3.25, 'PAYER', fontsize=9, color='#c0392b', fontweight='bold',
        rotation=90, va='center', ha='center')

# Dividing line
ax.axhline(6.5, color='#aaa', linewidth=1.5, linestyle='--', zorder=3)

# Nodes
box(ax, 5, 11, 2.8, 0.7, 'Coverage Verified\nPA Request Created', '#2980b9', actor='Provider')
box(ax, 5, 9.5, 2.4, 0.65, 'PA Request Submitted', '#2980b9', actor='Provider')

# Validation / Review
box(ax, 5, 8.0, 2.4, 0.65, 'PA Received &\nValidated', '#8e44ad', actor='Payer')
box(ax, 5, 6.5, 2.4, 0.65, 'PA Review Started', '#8e44ad', actor='Payer')

# Decision branch
box(ax, 2.2, 5.0, 2.0, 0.65, 'PA APPROVED ✓', '#27ae60', actor='Payer')
box(ax, 5.0, 5.0, 2.0, 0.65, 'PA PENDED', '#f39c12', actor='Payer')
box(ax, 7.8, 5.0, 2.0, 0.65, 'PA DENIED ✗', '#e74c3c', actor='Payer')

# Pended sub-path
box(ax, 5.0, 3.8, 2.5, 0.65, 'Add\'l Info Requested\n(Payer → Provider)', '#e67e22', actor='Payer')
box(ax, 5.0, 2.6, 2.5, 0.65, 'Documentation\nSubmitted (Provider)', '#2980b9', actor='Provider')
box(ax, 5.0, 1.4, 2.5, 0.65, 'Review Resumed →\nFinal Decision', '#8e44ad', actor='Payer')

# Appeal path
box(ax, 7.8, 3.5, 2.0, 0.65, 'Appeal Submitted\n+ Evidence', '#2980b9', actor='Provider')
box(ax, 7.8, 2.2, 2.0, 0.65, 'Denial Upheld or\nOverturned', '#8e44ad', actor='Payer')

# Arrows
arrow(ax, 5, 10.65, 5, 9.82)
arrow(ax, 5, 9.18, 5, 8.32)
arrow(ax, 5, 7.68, 5, 6.82)
# Decision
arrow(ax, 5, 6.18, 2.2, 5.32, 'Approved')
arrow(ax, 5, 6.18, 5.0, 5.32, 'Pended')
arrow(ax, 5, 6.18, 7.8, 5.32, 'Denied')
# Pended sub-path
arrow(ax, 5.0, 4.68, 5.0, 4.12)
arrow(ax, 5.0, 3.47, 5.0, 2.93)
arrow(ax, 5.0, 2.27, 5.0, 1.72)
# Appeal path
arrow(ax, 7.8, 4.68, 7.8, 3.82, 'Provider appeals')
arrow(ax, 7.8, 3.18, 7.8, 2.52)

# Note
ax.text(5, 0.3, "Note: All approval/denial logic is fictional. PAS provides operational state structure only.",
        ha='center', va='center', fontsize=8, color='#777', style='italic')

plt.tight_layout()
plt.savefig("projects/week 5/visualizations/visual_A_pas_workflow.png", dpi=130, bbox_inches='tight')
plt.close()
print("[OK] Visual A: PAS workflow diagram saved")


# ============================================================
# VISUAL B: Architecture Ladder
# ============================================================
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis('off')
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f8f9fa')
ax.set_title("Architecture Ladder: From Embedding-Only to Full Transformer",
             fontsize=13, fontweight='bold')

# Colors for each layer type
LAYER_COLORS = {
    'embed': '#3498db',
    'pe':    '#2980b9',
    'attn1': '#9b59b6',
    'attn4': '#8e44ad',
    'block': '#e74c3c',
    'head':  '#27ae60',
    'res':   '#f39c12',
    'ln':    '#1abc9c',
    'ffn':   '#e67e22',
}

def layer_box(ax, cx, cy, w, h, label, color='#3498db', fontsize=8):
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0.08", facecolor=color,
                           edgecolor='white', linewidth=1.2, zorder=5, alpha=0.9)
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha='center', va='center', fontsize=fontsize,
            color='white', fontweight='bold', zorder=6)

models = [
    ("Model A", 2.0, [
        ("Embedding\n+ Pos Enc", 'embed', 5.5),
        ("Linear Head\n(→ Vocab)", 'head', 3.5),
    ]),
    ("Model B", 5.5, [
        ("Embedding\n+ Pos Enc", 'embed', 5.5),
        ("1-Head\nCausal Attn", 'attn1', 4.5),
        ("Residual +", 'res', 3.8),
        ("Linear Head", 'head', 3.0),
    ]),
    ("Model C", 9.0, [
        ("Embedding\n+ Pos Enc", 'embed', 5.5),
        ("4-Head\nCausal Attn", 'attn4', 4.5),
        ("Residual +", 'res', 3.8),
        ("Linear Head", 'head', 3.0),
    ]),
    ("Model D", 13.0, [
        ("Embedding\n+ Pos Enc", 'embed', 5.5),
        ("LayerNorm\n→ 4-Head Attn\n→ Residual", 'block', 4.4),
        ("LayerNorm\n→ FFN 24→96→24\n→ Residual", 'ffn', 3.3),
        ("Block 2\n(same structure)", 'block', 2.2),
        ("Linear Head", 'head', 1.1),
    ]),
]

for model_name, cx, layers in models:
    ax.text(cx, 7.3, model_name, ha='center', va='center', fontsize=11,
            fontweight='bold', color='#2c3e50')
    
    prev_y = None
    for label, color_key, y in layers:
        layer_box(ax, cx, y, 2.8, 0.75, label, LAYER_COLORS[color_key], fontsize=8)
        if prev_y is not None:
            ax.annotate('', xy=(cx, y + 0.375), xytext=(cx, prev_y - 0.375),
                        arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))
        prev_y = y
    
    # What's new annotation
    if model_name == "Model B":
        ax.annotate("+ Single-head\ncausal attention", xy=(cx + 1.4, 4.5),
                    fontsize=7.5, color='#9b59b6', ha='left')
    elif model_name == "Model C":
        ax.annotate("+ 4 parallel\nattention heads", xy=(cx + 1.4, 4.5),
                    fontsize=7.5, color='#8e44ad', ha='left')
    elif model_name == "Model D":
        ax.annotate("+ LayerNorm,\nFFN, 2nd block,\nResidual", xy=(cx + 1.4, 3.5),
                    fontsize=7.5, color='#e74c3c', ha='left')

# Add-arrows between models
for x1, x2 in [(3.4, 4.1), (6.9, 7.6), (10.4, 11.1)]:
    ax.annotate('', xy=(x2, 6.5), xytext=(x1, 6.5),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.0))
    ax.text((x1+x2)/2, 6.7, '+', ha='center', va='center', fontsize=14,
            fontweight='bold', color='#27ae60')

# Legend
legend_elements = [
    mpatches.Patch(color=LAYER_COLORS['embed'], label='Embedding + PE'),
    mpatches.Patch(color=LAYER_COLORS['attn4'], label='Multi-Head Attention'),
    mpatches.Patch(color=LAYER_COLORS['ffn'], label='Feed-Forward Network'),
    mpatches.Patch(color=LAYER_COLORS['head'], label='Output Head'),
    mpatches.Patch(color=LAYER_COLORS['res'], label='Residual Connection'),
    mpatches.Patch(color=LAYER_COLORS['ln'], label='LayerNorm'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8, ncol=3)

plt.tight_layout()
plt.savefig("projects/week 5/visualizations/visual_B_architecture_ladder.png", dpi=130, bbox_inches='tight')
plt.close()
print("[OK] Visual B: Architecture ladder saved")


# ============================================================
# VISUAL C: Pre-LN Transformer Block
# ============================================================
fig, ax = plt.subplots(figsize=(6, 10))
ax.set_xlim(0, 6)
ax.set_ylim(0, 11)
ax.axis('off')
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f8f9fa')
ax.set_title("Pre-LN Transformer Block\n(Single block structure in Model D)",
             fontsize=12, fontweight='bold')

steps = [
    (3, 9.8, 2.8, 0.65, "x  (Input)", '#7f8c8d', 10),
    (3, 8.5, 2.8, 0.65, "LayerNorm₁", '#1abc9c', 9),
    (3, 7.2, 2.8, 0.75, "Causal Multi-Head\nAttention (4 heads)", '#8e44ad', 9),
    (3, 6.0, 2.8, 0.65, "x + attn_out\n(Residual skip-highway ①)", '#f39c12', 9),
    (3, 4.8, 2.8, 0.65, "LayerNorm₂", '#1abc9c', 9),
    (3, 3.5, 2.8, 0.75, "Feed-Forward Network\nReLU(x W₁ + b₁) W₂ + b₂", '#e67e22', 9),
    (3, 2.3, 2.8, 0.65, "x₁ + ffn_out\n(Residual skip-highway ②)", '#f39c12', 9),
    (3, 1.0, 2.8, 0.65, "x₂  (Block Output)", '#7f8c8d', 10),
]

prev_y = None
for cx, cy, w, h, label, color, fs in steps:
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0.1", facecolor=color,
                           edgecolor='white', linewidth=1.5, zorder=5, alpha=0.9)
    ax.add_patch(rect)
    ax.text(cx, cy, label, ha='center', va='center', fontsize=fs,
            color='white', fontweight='bold', zorder=6, multialignment='center')
    if prev_y is not None:
        ax.annotate('', xy=(cx, cy + h/2 + 0.05), xytext=(cx, prev_y - h/2 - 0.05),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
    prev_y = cy

# Draw residual skip arrows
# Residual 1: from input x (step 0) to step 3 (x + attn_out)
ax.annotate('', xy=(5.2, 6.0), xytext=(5.2, 9.8),
            arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2.5,
                            connectionstyle='arc3,rad=0'))
ax.text(5.5, 7.9, 'skip①\n(x unchanged)', fontsize=8, color='#e67e22', ha='center', va='center')

# Residual 2: from x₁ to x₁ + ffn_out
ax.annotate('', xy=(5.2, 2.3), xytext=(5.2, 6.0),
            arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2.5,
                            connectionstyle='arc3,rad=0'))
ax.text(5.5, 4.15, 'skip②\n(x₁ unchanged)', fontsize=8, color='#e67e22', ha='center', va='center')

# Shapes annotation on left
shape_notes = [
    (9.8, "(B, T, d_model)"),
    (8.5, "→ normalized"),
    (7.2, "(B, H, T, T) weights"),
    (6.0, "(B, T, d_model)"),
    (4.8, "→ normalized"),
    (3.5, "(B,T,d_ff) → (B,T,d_model)"),
    (2.3, "(B, T, d_model)"),
    (1.0, "(B, T, d_model)"),
]
for y, note in shape_notes:
    ax.text(0.25, y, note, fontsize=7, color='#555', ha='left', va='center')

plt.tight_layout()
plt.savefig("projects/week 5/visualizations/visual_C_transformer_block.png", dpi=130, bbox_inches='tight')
plt.close()
print("[OK] Visual C: Transformer block diagram saved")

print("\nAll static diagrams saved to projects/week 5/visualizations/")
