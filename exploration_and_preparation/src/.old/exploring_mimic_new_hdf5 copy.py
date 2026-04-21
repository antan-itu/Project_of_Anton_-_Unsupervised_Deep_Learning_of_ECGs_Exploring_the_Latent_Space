import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

FILE_PATH = '/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/exploration_and_preparation/logs/aligned_test_100.h5'
LOG_DIR = '/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/exploration_and_preparation/logs'

def find_indices(h5_filepath, target_diagnosis='atrial fibrillation', limit=25):
    """Searches the GT/Primary_Diagnosis array for specific strings."""
    print(f"\n--- Searching Dataset for '{target_diagnosis}' ---")
    indices = []
    
    with h5py.File(h5_filepath, 'r') as f:
        if 'GT/Primary_Diagnosis' not in f:
            print("Error: Primary_Diagnosis not found.")
            return []
            
        diagnoses = f['GT/Primary_Diagnosis'][:]
        
        for idx, diag_bytes in enumerate(diagnoses):
            diag_str = diag_bytes.decode('utf-8').lower()
            if target_diagnosis in diag_str:
                indices.append(idx)
                if len(indices) >= limit:
                    break
                    
    print(f"Found {len(indices)} matching cases.")
    return indices

def generate_all_12_lead_plots(h5_filepath, indices, base_save_dir, dataset_key='rhythm_filtered'):
    """Loops through all provided indices and saves individual 12-lead plots to a dedicated folder."""
    if not indices: 
        return
        
    # Create a dedicated subfolder so we don't clutter the main logs directory
    save_dir = os.path.join(base_save_dir, 'afib_12_lead_batch')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n--- Generating {len(indices)} Comprehensive 12-Lead Plots ---")
    print(f"Saving images to: {save_dir}\n")
    
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    with h5py.File(h5_filepath, 'r') as f:
        dataset = f[dataset_key]
        diagnoses_dataset = f['GT/Primary_Diagnosis']
        
        for count, idx in enumerate(indices):
            # Give a progress update in the terminal
            print(f"[{count+1}/{len(indices)}] Processing Index {idx}...")
            
            signal = dataset[idx]
            diagnosis = diagnoses_dataset[idx].decode('utf-8')
            
            if signal.shape[0] > signal.shape[1]:
                signal = signal.T 
                
            num_leads = signal.shape[0] 
            fig, axes = plt.subplots(num_leads, 1, figsize=(15, 16), sharex=True)
            fig.suptitle(f"12-Lead ECG | Index: {idx} | Diag: {diagnosis}", fontsize=14)
            
            for lead_idx in range(num_leads):
                ax = axes[lead_idx]
                ax.plot(signal[lead_idx], color='#1d3557', linewidth=0.8)
                
                label = lead_names[lead_idx] if lead_idx < 12 else f"Lead {lead_idx}"
                ax.set_ylabel(label, fontsize=10, rotation=0, labelpad=20, ha='right')
                
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_yticks([])
                
            axes[-1].set_xlabel('Time (Samples)', fontsize=12)
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            save_path = os.path.join(save_dir, f"aligned_12_lead_idx_{idx}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            # Close the figure so we don't run out of system RAM while generating 25 large plots
            plt.close()
            
    print("\nBatch generation complete!")

if __name__ == "__main__":
    # 1. Search for AFib cases
    afib_indices = find_indices(FILE_PATH, target_diagnosis='atrial fibrillation', limit=25)

    # 2. Generate and save all 25 individual 12-lead plots
    generate_all_12_lead_plots(FILE_PATH, afib_indices, LOG_DIR)