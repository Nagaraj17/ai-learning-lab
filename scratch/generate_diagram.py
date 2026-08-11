import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('topics/images', exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#0f172a')

for ax in (ax1, ax2):
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#475569')

# Standard Softmax Overflow
logits = [1000, 998, 995]
classes = ['Class 0', 'Class 1', 'Class 2']
ax1.bar(classes, [1000, 998, 995], color='#ef4444', alpha=0.8)
ax1.set_title('Standard Softmax (Naive)\nexp(1000) -> OVERFLOW (inf / NaN)', color='#f87171', fontsize=12, fontweight='bold')
ax1.set_ylabel('Raw Logits (z)', color='white')
for i, v in enumerate(logits):
    ax1.text(i, v/2, f'z = {v}\nexp({v}) -> inf!', ha='center', va='center', color='white', fontweight='bold', fontsize=10)

# Numerically Stable Softmax
shifted = np.array(logits) - np.max(logits)
exp_shifted = np.exp(shifted)
probs = exp_shifted / np.sum(exp_shifted)

bars = ax2.bar(classes, probs, color='#3b82f6', alpha=0.85)
ax2.set_title('Numerically Stable Softmax (Shifted)\nShifted Logits: z - max(z)', color='#60a5fa', fontsize=12, fontweight='bold')
ax2.set_ylabel('Softmax Probability P(x)', color='white')
ax2.set_ylim(0, 1.15)
for i, p in enumerate(probs):
    ax2.text(i, p + 0.04, f'Shifted z = {shifted[i]}\nP = {p:.4f}', ha='center', va='bottom', color='white', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('topics/images/numerical_stability_softmax.png', dpi=300, bbox_inches='tight')
print('Diagram generated successfully!')
