# Check for Dead Leads in the MIMIC-IV ECG Dataset

import h5py
import numpy as np
import pandas as pd
import os
import time
from multiprocessing import Pool, cpu_count

# File paths
FILE_PATH = '/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/data/MIMIC_IV_ECG_HDF5/mimic_iv_aligned_full.h5'
LOG_DIR = '/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/exploration_and_preparation/logs'

def scan_flatlines_worker(args):
    start_idx, end_idx = args
    
    with h5py.File(FILE_PATH, 'r') as f:
        # Load the chunk
        chunk = f['rhythm_filtered'][start_idx:end_idx]
        
        # Calculate Peak-to-Peak (Max - Min) - if the difference is exactly 0.0, the lead is completely flat.
        peak_to_peak = np.ptp(chunk, axis=1)
        
        # Check all 8 leads
        flatline_mask = (peak_to_peak == 0.0).any(axis=1)
        
        bad_relative_indices = np.where(flatline_mask)[0]
        bad_absolute_indices = bad_relative_indices + start_idx
        
    return bad_absolute_indices.tolist()

def locate_flatlines_parallel(h5_filepath, save_dir, chunk_size=50000):
    """Scans dataset and save the indices of affected ECGs."""
    print("\n--- Initializing Dead Lead Scan ---")
    start_time = time.time()
    
    with h5py.File(h5_filepath, 'r') as f:
        total_ecgs = f['rhythm_filtered'].shape[0]
        
    print(f"Total ECGs to scan: {total_ecgs}")
    
    tasks = []
    for i in range(0, total_ecgs, chunk_size):
        end_idx = min(i + chunk_size, total_ecgs)
        tasks.append((i, end_idx))
        
    num_cores = max(1, cpu_count() - 2)
    print(f"Using {num_cores} cores...")
    
    flatline_indices = []
    processed_count = 0
    
    with Pool(processes=num_cores) as pool:
        for result in pool.imap_unordered(scan_flatlines_worker, tasks):
            if result:
                flatline_indices.extend(result)
                
            processed_count += chunk_size
            current_progress = min(processed_count, total_ecgs)
            
            print(f"Scanned {current_progress}/{total_ecgs} | Flatlines found so far: {len(flatline_indices)}", end='\r')

    elapsed = time.time() - start_time
    
    print("\n\n--- Scan Complete ---")
    print(f"Total time: {elapsed/60:.2f} minutes")
    print(f"Total ECGs with dead leads found: {len(flatline_indices)}")
    
    if flatline_indices:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'flatline_ecg_indices.csv')
        
        flatline_indices.sort()
        df = pd.DataFrame({'flatline_index': flatline_indices})
        df.to_csv(save_path, index=False)
        print(f"Saved dead-lead list to:\n  {save_path}")

if __name__ == "__main__":
    locate_flatlines_parallel(FILE_PATH, LOG_DIR)