# The script visualizes ECGs with dead leads

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# File paths
FILE_PATH = '/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/data/MIMIC_IV_ECG_HDF5/mimic_iv_aligned_full.h5'
LOG_DIR = '/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/exploration_and_preparation/logs'

def visualize_corrupted_leads(h5_filepath, log_dir, num_to_plot=5):
    print("\n--- Visualizing Dead Lead ECGs ---")
    
    csv_path = os.path.join(log_dir, 'flatline_ecg_indices.csv')
    if not os.path.exists(csv_path):
        print(f"Error: Could not find the index CSV at {csv_path}")
        return
        
    # Load the indices
    df = pd.read_csv(csv_path)
    corrupted_indices = df['flatline_index'].values.tolist()
    
    if not corrupted_indices:
        print("No corrupted indices to plot.")
        return
        
    save_dir = os.path.join(log_dir, 'dead_lead_visualizations')
    os.makedirs(save_dir, exist_ok=True)
    
    # Plotting a subset of the ECGs
    target_indices = corrupted_indices[:num_to_plot]
    print(f"Generating plots for the first {len(target_indices)} ECGs...")
    
    lead_names = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    with h5py.File(h5_filepath, 'r') as f:
        dataset = f['rhythm_filtered']
        
        for count, idx in enumerate(target_indices):
            print(f"[{count+1}/{len(target_indices)}] Plotting Index {idx}...")
            
            signal = dataset[idx]
            
            if signal.shape[0] > signal.shape[1]:
                signal = signal.T 
                
            num_leads = signal.shape[0] 
            fig, axes = plt.subplots(num_leads, 1, figsize=(15, 12), sharex=True)
            fig.suptitle(f"Bad 8-Lead ECG | Index: {idx}", fontsize=14)
            
            for lead_idx in range(num_leads):
                ax = axes[lead_idx]
                lead_signal = signal[lead_idx]
                
                # Check if the specific lead is the dead one (Peak-to-Peak == 0)
                is_flatline = np.ptp(lead_signal) == 0.0
                
                # Change color to red for dead leads
                line_color = '#e63946' if is_flatline else '#1d3557'
                label_color = 'red' if is_flatline else 'black'
                label_weight = 'bold' if is_flatline else 'normal'
                
                ax.plot(lead_signal, color=line_color, linewidth=1.5 if is_flatline else 0.8)
                
                label_text = f"Lead {lead_names[lead_idx]}"
                if is_flatline:
                    label_text += "\n[DEAD]"
                    
                ax.set_ylabel(label_text, fontsize=10, rotation=0, labelpad=25, ha='right', 
                              color=label_color, weight=label_weight)
                
                ax.grid(True, linestyle='--', alpha=0.5)
                
                # Add vertical padding for better visibility
                if is_flatline:
                    ax.set_ylim(-0.1, 0.1)
                else:
                    ax.set_yticks([])
                
            axes[-1].set_xlabel('Time (Samples)', fontsize=12)
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            save_path = os.path.join(save_dir, f"dead_lead_idx_{idx}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
    print(f"\nPlots saved to:\n  {save_dir}")

if __name__ == "__main__":
    visualize_corrupted_leads(FILE_PATH, LOG_DIR, num_to_plot=5)