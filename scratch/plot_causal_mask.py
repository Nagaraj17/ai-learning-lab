import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('topics/images', exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='#0f172a')

tokens = ["A", "Cat", "sat", "on", "a", "mat"]
T = len(tokens)

# --- Left: Unmasked Attention (Cheating Allowed) ---
unmasked_A = np.ones((T, T)) / T

ax1.set_facecolor('#1e293b')
im1 = ax1.imshow(unmasked_A, cmap='Reds', vmin=0, vmax=0.5)
ax1.set_xticks(range(T))
ax1.set_yticks(range(T))
ax1.set_xticklabels(tokens, color='white', fontsize=11)
ax1.set_yticklabels(tokens, color='white', fontsize=11)
ax1.set_title('Unmasked Attention (CHEATING!)\nRow "a" can see "mat"', color='#ef4444', fontsize=12, fontweight='bold', pad=12)
ax1.set_xlabel('Key Token (K)', color='#cbd5e1')
ax1.set_ylabel('Query Token (Q)', color='#cbd5e1')

# Highlight cheating cell (Row 4 "a", Col 5 "mat")
rect = plt.Rectangle((4.5, 3.5), 1, 1, fill=False, edgecolor='#f59e0b', linewidth=3)
ax1.add_patch(rect)
ax1.text(4.7, 4.2, 'PEEK!', color='#f59e0b', fontweight='bold', fontsize=10)

# --- Right: Causally Masked Attention (Safe Training) ---
mask = np.tril(np.ones((T, T)))
masked_A = np.zeros((T, T))
for i in range(T):
    masked_A[i, :i+1] = 1.0 / (i + 1)

ax2.set_facecolor('#1e293b')
im2 = ax2.imshow(masked_A, cmap='Blues', vmin=0, vmax=0.5)
ax2.set_xticks(range(T))
ax2.set_yticks(range(T))
ax2.set_xticklabels(tokens, color='white', fontsize=11)
ax2.set_yticklabels(tokens, color='white', fontsize=11)
ax2.set_title('Causally Masked Attention (SAFE)\nUpper Triangle = 0% (Blocked)', color='#10b981', fontsize=12, fontweight='bold', pad=12)
ax2.set_xlabel('Key Token (K)', color='#cbd5e1')
ax2.set_ylabel('Query Token (Q)', color='#cbd5e1')

# Mark blocked upper triangle with 'X'
for i in range(T):
    for j in range(i+1, T):
        ax2.text(j, i, '0', color='#64748b', ha='center', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('topics/images/causal_masking_cheating_diagram.png', dpi=150)
print("Saved causal masking diagram to topics/images/causal_masking_cheating_diagram.png")
