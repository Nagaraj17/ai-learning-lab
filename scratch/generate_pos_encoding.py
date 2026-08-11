import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('topics/images', exist_ok=True)

def get_sinusoidal_positional_encoding(seq_len, d_model):
    P = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            denom = 10000 ** (i / d_model)
            P[pos, i] = np.sin(pos / denom)
            if i + 1 < d_model:
                P[pos, i + 1] = np.cos(pos / denom)
    return P

# 1. Print small numbers for T=4, d_model=4
T_small = 4
d_small = 4
P_small = get_sinusoidal_positional_encoding(T_small, d_small)

print("--- NUMERICAL VECTORS (T=4, d_model=4) ---")
for pos in range(T_small):
    vec_str = ", ".join([f"{v:+.4f}" for v in P_small[pos]])
    print(f"Position {pos} vector p_{pos}: [{vec_str}]")

# 2. Generate Heatmap Plot for T=20, d_model=64
T_large = 20
d_large = 64
P_large = get_sinusoidal_positional_encoding(T_large, d_large)

plt.figure(figsize=(10, 5))
plt.imshow(P_large, cmap='magma', aspect='auto')
plt.colorbar(label='Value')
plt.title('Sinusoidal Positional Encoding Heatmap P (T=20, d_model=64)', fontsize=12)
plt.xlabel('Vector Dimension (0 to 63)', fontsize=10)
plt.ylabel('Sequence Position (0 to 19)', fontsize=10)
plt.tight_layout()
plt.savefig('topics/images/positional_encoding_heatmap.png', dpi=150)
print("Heatmap saved to topics/images/positional_encoding_heatmap.png")
