import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Define absolute paths for the HDF5 dataset and the output logging directory.
FILE_PATH = '/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/data/full_training_set/training_dataset.h5'
LOG_DIR = '/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/exploration_and_preparation/logs'

def explore_h5_structure(filepath, save_dir):
    """
    Recursively maps the hierarchical structure of the HDF5 file
    and saves the output to a text file in the logging directory.
    """
    print(f"\n--- Mapping HDF5 Structure ---")
    
    output_lines = [f"--- HDF5 Structure: {filepath.split('/')[-1]} ---", "Root: '/'"]
    
    def append_hierarchy(name, obj):
        indent = (name.count('/') + 1) * '  '
        if isinstance(obj, h5py.Dataset):
            output_lines.append(f"{indent}Dataset: '{name}' | Shape: {obj.shape} | Dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            output_lines.append(f"{indent}Group: '{name}'")
            
    with h5py.File(filepath, 'r') as f:
        f.visititems(append_hierarchy)
        
    output_lines.append("-" * 50)
    output_text = "\n".join(output_lines)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "h5_structure_map.txt")
    
    with open(save_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(output_text)
        
    print(f"Structure map successfully saved to:\n   {save_path}")

def export_metadata_to_csv(filepath, save_dir, num_rows=100):
    """
    Extracts the ground truth metadata table
    """
    print(f"\n--- Extracting Top {num_rows} Metadata Rows to CSV ---")
    
    with h5py.File(filepath, 'r') as f:
        if 'GT/table' not in f:
            print("Warning: Dataset 'GT/table' not found in the file.")
            return
            
        raw_data = f['GT/table'][:num_rows]
        
        data_dict = {}
        for col_name in raw_data.dtype.names:
            col_data = raw_data[col_name]
            if col_data.ndim > 1:
                data_dict[col_name] = list(col_data)
            else:
                data_dict[col_name] = col_data
                
        df = pd.DataFrame(data_dict)
        
        for col in df.columns:
            df[col] = df[col].apply(
                lambda x: x[0].decode('utf-8') if isinstance(x, np.ndarray) and len(x) == 1 and isinstance(x[0], bytes)
                else (x.decode('utf-8') if isinstance(x, bytes) else x)
            )
        
        column_mapping = {
            'index': 'Patient_ID',
            'values_block_1': 'Timestamp',
            'values_block_2': 'Primary_Diagnosis',
            'values_block_3': 'Secondary_Diagnosis',
            'values_block_4': 'Notes',
            'values_block_21': 'Frequency_Filter',
            'values_block_22': 'Baseline_Filter'
        }
        df = df.rename(columns=column_mapping)
        df = df.dropna(axis=1, how='all')
        
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, 'mimic_metadata_top100.csv')
        df.to_csv(csv_path, index=False)
        print(f"Metadata successfully saved to:\n   {csv_path}")

def plot_ecg_sample(filepath, dataset_key, sample_index, save_dir):
    """
    Extracts a single multi-lead ECG sample from the specified dataset,
    selects the first lead, plots the waveform, and saves the figure.
    """
    print(f"\n--- Plotting Single ECG Sample (Index: {sample_index}) ---")
    with h5py.File(filepath, 'r') as f:
        if dataset_key not in f:
            print(f"Error: Dataset key '{dataset_key}' not found.")
            return
            
        dataset = f[dataset_key]
        sample = dataset[sample_index]
        
        plt.figure(figsize=(15, 4))
        
        if len(sample.shape) > 1:
            if sample.shape[0] > sample.shape[1]:
                signal_to_plot = sample[:, 0] 
            else:
                signal_to_plot = sample[0, :]
            title = f'ECG Signal | Dataset: {dataset_key} | Sample: {sample_index} | Lead 0'
        else:
            signal_to_plot = sample
            title = f'ECG Signal | Dataset: {dataset_key} | Sample: {sample_index}'

        plt.title(title)
        plt.plot(signal_to_plot, color='#e63946', linewidth=1)
        plt.xlabel('Time (Samples)')
        plt.ylabel('Amplitude (mV)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        os.makedirs(save_dir, exist_ok=True)
        filename = f"ecg_{dataset_key.replace('/', '_')}_sample_{sample_index}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot successfully saved to:\n   {save_path}")
            
        plt.close()

def find_indices(h5_filepath, target_diagnosis='atrial fibrillation', chunk_size=50000, limit=25):
    """
    Efficiently searches the entire HDF5 metadata table for indices 
    that correspond to variations of a specific target diagnosis.
    """
    print(f"\n--- Searching Full Dataset for '{target_diagnosis}' Indices ---")
    
    indices = []
    with h5py.File(h5_filepath, 'r') as f:
        if 'GT/table' not in f or 'values_block_2' not in f['GT/table'].dtype.names:
            print("Warning: Necessary metadata column not found.")
            return []
            
        total_rows = len(f['GT/table'])
        for i in range(0, total_rows, chunk_size):
            end_idx = min(i + chunk_size, total_rows)
            chunk_diag = f['GT/table']['values_block_2'][i:end_idx] 
            
            for j, diag_array in enumerate(chunk_diag):
                try:
                    if len(diag_array) > 0 and isinstance(diag_array[0], bytes):
                        diag_str = diag_array[0].decode('utf-8').lower().strip()
                        if diag_str.startswith(target_diagnosis.lower()):
                            indices.append(i + j)
                            if len(indices) >= limit:
                                break
                except UnicodeDecodeError:
                    continue
            
            if len(indices) >= limit:
                break
                
        print(f"Found {len(indices)} matching ECGs.")
        return indices

def plot_and_save_ecg_batch(h5_filepath, indices, save_dir, dataset_key='rhythm_filtered'):
    """
    Loads ECG signals based on indices, creates a batch plot with subplots,
    and saves the combined figure to the log directory.
    """
    if not indices:
        print("No matching indices found to plot.")
        return
        
    print(f"\n--- Batch Plotting {len(indices)} ECGs ---")
    
    fig = plt.figure(figsize=(20, 16))
    grid_rows, grid_cols = 5, 5 
    
    with h5py.File(h5_filepath, 'r') as f:
        if dataset_key not in f:
            print(f"Error: Dataset key '{dataset_key}' not found.")
            plt.close() 
            return
            
        dataset = f[dataset_key]
        
        for i, idx in enumerate(indices):
            if i >= 25: break 
            
            signal = dataset[idx]
            
            if signal.shape[0] > signal.shape[1]:
                signal_1d = signal[:, 0]
            else:
                signal_1d = signal[0, :]
                
            ax = fig.add_subplot(grid_rows, grid_cols, i + 1)
            ax.plot(signal_1d, color='#e63946', linewidth=0.5) 
            ax.set_title(f"Idx: {idx}", fontsize=10)
            ax.set_xticks([]) 
            ax.tick_params(labelsize=8) 
            
    fig.suptitle("Batch Plot: Filtered Cases", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "batch_plot_filtered_cases.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Batch plot successfully saved to:\n   {save_path}")
        
    plt.close()

def plot_all_leads(h5_filepath, idx, save_dir, dataset_key='rhythm_filtered'):
    """
    Extracts a specific ECG and plots all 8 leads stacked vertically 
    to properly simulate a clinical ECG readout.
    """
    with h5py.File(h5_filepath, 'r') as f:
        signal = f[dataset_key][idx]
        
        # Ensure shape is (8 leads, 5000 time steps)
        if signal.shape[0] > signal.shape[1]:
            signal = signal.T 
            
        num_leads = signal.shape[0]
        
        # Create a tall figure to stack the leads
        fig, axes = plt.subplots(num_leads, 1, figsize=(15, 12), sharex=True)
        fig.suptitle(f"Full 8-Lead ECG | Index: {idx} | Dataset: {dataset_key}", fontsize=16)
        
        for lead_idx in range(num_leads):
            ax = axes[lead_idx]
            ax.plot(signal[lead_idx], color='#1d3557', linewidth=0.8)
            ax.set_ylabel(f"Lead {lead_idx}", fontsize=10, rotation=0, labelpad=30, ha='right')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_yticks([])
            
        axes[-1].set_xlabel('Time (Samples)', fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"full_8_lead_ecg_idx_{idx}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive 8-lead plot for index {idx} to:\n   {save_path}")
        plt.close()

if __name__ == "__main__":
    # 1. Map out the HDF5 structure and save it to a text file
    explore_h5_structure(filepath=FILE_PATH, save_dir=LOG_DIR)
    
    # 2. Export the top 100 rows of metadata to the logs folder as a CSV
    export_metadata_to_csv(filepath=FILE_PATH, save_dir=LOG_DIR, num_rows=100)
    
    # 3. Generate and save a plot for the first sample in the filtered dataset
    plot_ecg_sample(filepath=FILE_PATH, dataset_key='rhythm_filtered', sample_index=0, save_dir=LOG_DIR)
    
    # 4. Find up to 25 indices for 'atrial fibrillation' cases
    afib_indices = find_indices(h5_filepath=FILE_PATH, target_diagnosis='atrial fibrillation', limit=25)
    
    # 5. Batch plot and save the ECG waveforms for the found indices (Lead 0 only)
    plot_and_save_ecg_batch(h5_filepath=FILE_PATH, indices=afib_indices, save_dir=LOG_DIR, dataset_key='rhythm_filtered')

    # 6. Generate comprehensive 8-lead plots for the first 3 AFib cases to check clinical alignment
    print("\n--- Generating Comprehensive 8-Lead Plots ---")
    if afib_indices:
        for index in afib_indices[:3]:
            plot_all_leads(h5_filepath=FILE_PATH, idx=index, save_dir=LOG_DIR, dataset_key='rhythm_filtered')
    else:
        print("No matching cases found to generate 8-lead plots.")
    
    print("\n--- All exploration tasks complete. Check the logs directory. ---")