import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/data/full_training_set/training_dataset.h5"

print("Loading Data for Visual Check...")
df_gt = pd.read_hdf(FILE_PATH, key='GT')

# --- THE CRITICAL FIX ---
df_gt = df_gt.sort_values(by='h5idx').reset_index(drop=True)

# Find Normal and AFib indices
target_list = ["ATRIAL FIBRILLATION", "Atrial fibrillation", "Atrial fibrillation."]
mask = df_gt[[f'report_{i}' for i in range(18)]].fillna('').astype(str).apply(lambda row: any(row.isin(target_list)), axis=1)

# Randomly sample 25 of each for a robust visual check
print("Sampling 25 random Normal and 25 random AFib cases...")
afib_indices = mask[mask == True].sample(n=25, random_state=42).index.tolist()
normal_indices = mask[mask == False].sample(n=25, random_state=42).index.tolist()

print("Extracting waveforms from HDF5...")
with h5py.File(FILE_PATH, 'r') as h5f:
    # Load the 25 AFib and 25 Normal ECGs (Lead I)
    afib_ecgs = [h5f['rhythm_filtered'][idx, :, 0] for idx in afib_indices]
    normal_ecgs = [h5f['rhythm_filtered'][idx, :, 0] for idx in normal_indices]

# ---------------------------------------------------------
# Plot 1: 25 NORMAL ECGs
# ---------------------------------------------------------
print("Generating Normal ECG grid...")
fig_norm, axes_norm = plt.subplots(5, 5, figsize=(25, 15))
fig_norm.suptitle("25 NORMAL ECGs (Visual Alignment Check)", fontsize=22, fontweight='bold')

for i, ax in enumerate(axes_norm.flatten()):
    ax.plot(normal_ecgs[i], color='blue', linewidth=0.8)
    ax.set_title(f"Label: Normal | Index: {normal_indices[i]}", fontsize=10)
    ax.set_xticks([]) # Hide ticks for a cleaner visual scan
    ax.set_yticks([])

plt.tight_layout()
fig_norm.subplots_adjust(top=0.92)
plt.savefig("alignment_check_25_normal.png", dpi=200)
plt.close(fig_norm)

# ---------------------------------------------------------
# Plot 2: 25 AFIB ECGs
# ---------------------------------------------------------
print("Generating AFib ECG grid...")
fig_afib, axes_afib = plt.subplots(5, 5, figsize=(25, 15))
fig_afib.suptitle("25 ATRIAL FIBRILLATION ECGs (Visual Alignment Check)", fontsize=22, fontweight='bold', color='darkred')

for i, ax in enumerate(axes_afib.flatten()):
    ax.plot(afib_ecgs[i], color='red', linewidth=0.8)
    ax.set_title(f"Label: AFib | Index: {afib_indices[i]}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
fig_afib.subplots_adjust(top=0.92)
plt.savefig("alignment_check_25_afib.png", dpi=200)
plt.close(fig_afib)

print("\nSaved 'alignment_check_25_normal.png' and 'alignment_check_25_afib.png'.")
