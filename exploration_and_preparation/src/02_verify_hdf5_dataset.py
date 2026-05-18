# This script verifies the integrity of the HDF5 dataset by creating a batch of 8-lead ECG plots

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

FILE_PATH = '/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/data/MIMIC_IV_ECG_HDF5/mimic_iv_aligned_full.h5'
LOG_DIR = '/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/exploration_and_preparation/logs'

def find_indices(h5_filepath, target_diagnosis='atrial fibrillation', limit=25):
    print(f"\n--- Searching Dataset across clinical reports for '{target_diagnosis}' ---")
    indices = []
    
    with h5py.File(h5_filepath, 'r') as f:
        gt_group = f['GT']
        
        # Find columns that start with 'report_'
        report_keys = [key for key in gt_group.keys() if key.startswith('report_')]
        
        # Load the text arrays into memory
        reports_data = [gt_group[key][:] for key in report_keys]
        total_rows = len(reports_data[0])
        
        for idx in range(total_rows):
            # Check every column for the specific patient
            for col_data in reports_data:
                diag_str = col_data[idx].decode('utf-8').lower()
                if target_diagnosis in diag_str:
                    indices.append(idx)
                    break
            
            if len(indices) >= limit:
                break
                    
    print(f"Found {len(indices)} matching cases.")
    return indices

def generate_8_lead_plots(h5_filepath, indices, base_save_dir):
    if not indices: 
        return
        
    save_dir = os.path.join(base_save_dir, 'afib_8_lead_batch')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n--- Generating 8-Lead ECG Plots ---")
    
    lead_names = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    with h5py.File(h5_filepath, 'r') as f:
        dataset = f['rhythm_filtered']
        gt_group = f['GT']
        report_keys = sorted([key for key in gt_group.keys() if key.startswith('report_')])
        
        for count, idx in enumerate(indices):
            print(f"[{count+1}/{len(indices)}] Saving plot for Index {idx}...")
            
            signal = dataset[idx]
            
            # Putting the diagnosis names in the title
            patient_diags = []
            for key in report_keys:
                diag = gt_group[key][idx].decode('utf-8')
                if diag:
                    patient_diags.append(diag)
            
            title_diagnosis = " | ".join(patient_diags[:2])
            
            if signal.shape[0] > signal.shape[1]:
                signal = signal.T 
                
            num_leads = signal.shape[0] 
            fig, axes = plt.subplots(num_leads, 1, figsize=(15, 12), sharex=True)
            fig.suptitle(f"8-Lead Independent ECG | Index: {idx}\nDiag: {title_diagnosis}", fontsize=14)
            
            for lead_idx in range(num_leads):
                ax = axes[lead_idx]
                ax.plot(signal[lead_idx], color='#1d3557', linewidth=0.8)
                
                label = lead_names[lead_idx] if lead_idx < len(lead_names) else f"Lead {lead_idx}"
                ax.set_ylabel(label, fontsize=10, rotation=0, labelpad=20, ha='right')
                
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_yticks([])
                
            axes[-1].set_xlabel('Time (Samples)', fontsize=12)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            save_path = os.path.join(save_dir, f"verified_8_lead_idx_{idx}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    afib_indices = find_indices(FILE_PATH, target_diagnosis='atrial fibrillation', limit=25)
    generate_8_lead_plots(FILE_PATH, afib_indices, LOG_DIR)