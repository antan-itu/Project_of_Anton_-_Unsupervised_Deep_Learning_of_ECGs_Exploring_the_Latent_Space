# Extracting 400 random ECGs from the holdout set for manual review.

import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# --- 1. SET DIRECTORIES ---
BASE_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space"
DATA_PATH = os.path.join(BASE_DIR, "data/MIMIC_IV_ECG_HDF5/mimic_iv_holdout.h5")

# Creating a folder for the files
OUTPUT_DIR = os.path.join(BASE_DIR, "exploration_and_preparation/logs/manual_review_400_ecgs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

# Selecting 400 random ECGs 
NUM_SAMPLES = 400
np.random.seed(42) 

print(f"Opening HDF5 file: {DATA_PATH}")
with h5py.File(DATA_PATH, 'r') as f:
    dataset = f['rhythm_filtered']
    total_ecgs = dataset.shape[0]
    print(f"Total ECGs in holdout: {total_ecgs:,}")
    
    random_indices = np.random.choice(total_ecgs, NUM_SAMPLES, replace=False)
    # Sort them to make manual tracking easier
    random_indices.sort()

    # Generating and saving the plots
    print("\n--- Generating 400 Random 8-Lead Plots ---")
    lead_names = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    for count, idx in enumerate(random_indices):
        if (count + 1) % 25 == 0 or count == 0:
            print(f"[{count+1:03d}/{NUM_SAMPLES}] Saving plot for Index {idx}...")
            
        signal = dataset[idx]
        
        if signal.shape[0] > signal.shape[1]:
            signal = signal.T 
            
        fig, axes = plt.subplots(8, 1, figsize=(15, 12), sharex=True)
        
        fig.suptitle(f"HDF5 Index: {idx}", fontsize=16, fontweight='bold')
        
        for lead_idx in range(8):
            ax = axes[lead_idx]
            ax.plot(signal[lead_idx], color='#1d3557', linewidth=0.8)
            
            label = lead_names[lead_idx]
            ax.set_ylabel(label, fontsize=12, rotation=0, labelpad=20, ha='right', fontweight='bold')
            
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_yticks([])
            
        axes[-1].set_xlabel('Time (Samples)', fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save with zero-padding
        save_path = os.path.join(PLOT_DIR, f"ecg_idx_{idx:07d}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

# Creating a CSV for manual review
print("\n--- Exporting Review CSV ---")
csv_path = os.path.join(OUTPUT_DIR, "manual_afib_review.csv")

# Adding a column for labeling
review_df = pd.DataFrame({
    'HDF5_Index': random_indices,
    'Manual_AFib_Label': ''  # Suggestion: Fill with 1 for AFib, 0 for Normal/Other
})

review_df.to_csv(csv_path, index=False)
print(f"Batch and CSV saved to: {OUTPUT_DIR}")