import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('topics/images', exist_ok=True)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), facecolor='#0f172a')

# Set dark theme background
for ax in (ax1, ax2, ax3):
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#475569')

# --- Panel 1: Bounded Values vs Linear Explosion ---
pos = np.linspace(0, 100, 500)
sine_wave = np.sin(pos / 5)
linear = pos / 10

ax1.plot(pos, sine_wave, label='Sinusoidal (-1 to +1)', color='#06b6d4', linewidth=2.5)
ax1.plot(pos, linear, label='Raw Integer Index (Explodes!)', color='#ef4444', linestyle='--', linewidth=2)
ax1.axhline(1.0, color='#94a3b8', linestyle=':', alpha=0.6)
ax1.axhline(-1.0, color='#94a3b8', linestyle=':', alpha=0.6)
ax1.set_title('1. Bounded Values (-1 to +1)', color='white', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Sequence Position (pos)', color='#cbd5e1')
ax1.set_ylabel('Encoding Value', color='#cbd5e1')
ax1.set_ylim(-2, 10)
ax1.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='white')
ax1.grid(True, linestyle=':', alpha=0.2, color='#94a3b8')

# --- Panel 2: Extrapolation (512 to 1000) ---
pos_train = np.linspace(0, 512, 500)
pos_test = np.linspace(512, 1000, 500)

wave_train = np.sin(pos_train / 30)
wave_test = np.sin(pos_test / 30)

ax2.plot(pos_train, wave_train, label='Trained Range (0 to 512)', color='#10b981', linewidth=2.5)
ax2.plot(pos_test, wave_test, label='Extrapolated Unseen (512 to 1000)', color='#f59e0b', linestyle='-', linewidth=2.5)
ax2.axvline(512, color='#ef4444', linestyle='--', label='Training Limit (512)')
ax2.set_title('2. Extrapolation to Unseen Lengths', color='white', fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Sequence Position (pos)', color='#cbd5e1')
ax2.set_ylabel('Encoding Value', color='#cbd5e1')
ax2.set_ylim(-1.5, 1.5)
ax2.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='white', fontsize=9)
ax2.grid(True, linestyle=':', alpha=0.2, color='#94a3b8')

# --- Panel 3: Relative Distance via Trigonometric Rotation ---
theta = np.linspace(0, 2*np.pi, 200)
r = 1.0
x = r * np.cos(theta)
y = r * np.sin(theta)

ax3.plot(x, y, color='#94a3b8', linestyle='--', alpha=0.7)

pos1_angle = np.pi / 6  # 30 deg
pos2_angle = np.pi / 6 + np.pi / 3  # 30 + 60 = 90 deg

ax3.plot([0, np.cos(pos1_angle)], [0, np.sin(pos1_angle)], color='#ec4899', linewidth=2.5, marker='o', label='Position i')
ax3.plot([0, np.cos(pos2_angle)], [0, np.sin(pos2_angle)], color='#8b5cf6', linewidth=2.5, marker='o', label='Position i + k')

# Arc for relative angle k
arc_theta = np.linspace(pos1_angle, pos2_angle, 50)
ax3.plot(0.4 * np.cos(arc_theta), 0.4 * np.sin(arc_theta), color='#f43f5e', linewidth=2)
ax3.text(0.45, 0.45, r'Offset $\Delta k$', color='#f43f5e', fontsize=11, fontweight='bold')

ax3.set_title('3. Relative Distance via Rotation', color='white', fontsize=12, fontweight='bold', pad=10)
ax3.set_xlim(-1.3, 1.3)
ax3.set_ylim(-1.3, 1.3)
ax3.set_aspect('equal')
ax3.axis('off')
ax3.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='white', loc='lower right')

plt.tight_layout()
plt.savefig('topics/images/positional_encoding_3_properties.png', dpi=150)
print("Saved 3-panel visualization to topics/images/positional_encoding_3_properties.png")
