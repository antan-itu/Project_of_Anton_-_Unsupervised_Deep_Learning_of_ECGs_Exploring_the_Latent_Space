import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

FILE_PATH = '/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/exploration_and_preparation/logs/aligned_test_100.h5'
LOG_DIR = '/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/exploration_and_preparation/logs'

def find_indices(h5_filepath, target_diagnosis='atrial fibrillation', limit=25):
    """Searches the new GT/Primary_Diagnosis array for specific strings."""
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

def plot_and_save_ecg_batch(h5_filepath, indices, save_dir, dataset_key='rhythm_filtered'):
    """Plots Lead II for a batch of ECGs."""
    if not indices: return
    
    print(f"--- Batch Plotting {len(indices)} AFib Cases (Lead II) ---")
    fig = plt.figure(figsize=(20, 16))
    grid_rows, grid_cols = 5, 5 
    
    with h5py.File(h5_filepath, 'r') as f:
        dataset = f[dataset_key]
        
        for i, idx in enumerate(indices):
            if i >= 25: break 
            signal = dataset[idx]
            
            # Extract Lead II 
            signal_1d = signal[:, 1] if signal.shape[0] > signal.shape[1] else signal[1, :]
                
            ax = fig.add_subplot(grid_rows, grid_cols, i + 1)
            ax.plot(signal_1d, color='#e63946', linewidth=0.5) 
            ax.set_title(f"Idx: {idx}", fontsize=10)
            ax.set_xticks([]) 
            ax.tick_params(labelsize=8) 
            
    fig.suptitle("Batch Plot: AFib Cases (Lead II)", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "aligned_afib_batch_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved batch plot to:\n   {save_path}")
    plt.close()

def plot_all_12_leads(h5_filepath, idx, save_dir, dataset_key='rhythm_filtered'):
    """Extracts a specific ECG and plots all 12 leads stacked vertically."""
    with h5py.File(h5_filepath, 'r') as f:
        signal = f[dataset_key][idx]
        diagnosis = f['GT/Primary_Diagnosis'][idx].decode('utf-8')
        
        if signal.shape[0] > signal.shape[1]:
            signal = signal.T 
            
        num_leads = signal.shape[0] # Will be 12
        fig, axes = plt.subplots(num_leads, 1, figsize=(15, 16), sharex=True)
        fig.suptitle(f"12-Lead ECG | Index: {idx} | Diag: {diagnosis}", fontsize=14)
        
        # Standard order for MIMIC 12-lead (I, II, III, aVR, aVL, aVF, V1-V6)
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        for lead_idx in range(num_leads):
            ax = axes[lead_idx]
            ax.plot(signal[lead_idx], color='#1d3557', linewidth=0.8)
            
            # Label the y-axis with the standard clinical lead name
            label = lead_names[lead_idx] if lead_idx < 12 else f"Lead {lead_idx}"
            ax.set_ylabel(label, fontsize=10, rotation=0, labelpad=20, ha='right')
            
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_yticks([])
            
        axes[-1].set_xlabel('Time (Samples)', fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"aligned_12_lead_idx_{idx}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 12-lead plot to:\n   {save_path}")
        plt.close()

if __name__ == "__main__":
    # 1. Search for AFib cases using the newly extracted clinical text
    afib_indices = find_indices(FILE_PATH, target_diagnosis='atrial fibrillation')
    
    # 2. Batch plot Lead II
    plot_and_save_ecg_batch(FILE_PATH, afib_indices, LOG_DIR)

    # 3. Generate the 12-lead plots for the first 3 cases
    if afib_indices:
        print("\n--- Generating Comprehensive 12-Lead Plots ---")
        for index in afib_indices[:3]:
            plot_all_12_leads(FILE_PATH, index, LOG_DIR)