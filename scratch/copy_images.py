import os
import shutil

brain_dir = r"C:\Users\Nagar\.gemini\antigravity-ide\brain\d8fbe819-14a7-477b-ac2f-108a287c9de3"
target_dir = r"c:\Users\Nagar\source\repos\ai-learning-lab\topics\images"

os.makedirs(target_dir, exist_ok=True)

# Copy positional encoding infographic
img1_src = os.path.join(brain_dir, "positional_encoding_3_properties_infographic_1786041232811.png")
img1_dst = os.path.join(target_dir, "positional_encoding_3_properties_infographic.png")

if os.path.exists(img1_src):
    shutil.copy2(img1_src, img1_dst)
    print(f"Copied {img1_src} -> {img1_dst}")

# Copy causal masking cheating infographic
img2_src = os.path.join(brain_dir, "causal_masking_cheating_infographic_1786042058784.png")
img2_dst = os.path.join(target_dir, "causal_masking_cheating_infographic.png")

if os.path.exists(img2_src):
    shutil.copy2(img2_src, img2_dst)
    print(f"Copied {img2_src} -> {img2_dst}")

print("Image copy process complete!")
