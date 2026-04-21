# This script checks for missing and corrupted data in the HDF5 file.

import h5py
import numpy as np
import pandas as pd
import os
import time
from multiprocessing import Pool, cpu_count

# File paths
FILE_PATH = '/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/data/MIMIC_IV_ECG_HDF5/mimic_iv_aligned_full.h5'
LOG_DIR = '/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/exploration_and_preparation/logs'

def scan_chunk_worker(args):
    """
    Scan chunks in parallel to identify to check for NaNs or Infinities in the ECG data.
    """
    start_idx, end_idx = args
    
    # Opening the file
    with h5py.File(FILE_PATH, 'r') as f:
        chunk = f['rhythm_filtered'][start_idx:end_idx]
        
        # Check for NaNs or Infinities
        bad_mask = np.isnan(chunk).any(axis=(1, 2)) | np.isinf(chunk).any(axis=(1, 2))
        
        bad_relative_indices = np.where(bad_mask)[0]
        bad_absolute_indices = bad_relative_indices + start_idx
        
    return bad_absolute_indices.tolist()

def locate_corrupted_ecgs_parallel(h5_filepath, save_dir, chunk_size=20000):
    print("\n--- Initializing Parallel Missing Data Scan ---")
    start_time = time.time()
    
    # Total number of ECGs in the dataset
    with h5py.File(h5_filepath, 'r') as f:
        total_ecgs = f['rhythm_filtered'].shape[0]
        
    print(f"Total ECGs to scan: {total_ecgs}")
    
    # Finding indices for parallel processing
    tasks = []
    for i in range(0, total_ecgs, chunk_size):
        end_idx = min(i + chunk_size, total_ecgs)
        tasks.append((i, end_idx))
        
    # Selecting the number of cores
    num_cores = max(1, cpu_count() - 2)
    print(f"Spinning up {num_cores} parallel worker processes...")
    print(f"Batch size per worker: {chunk_size}\n")
    
    corrupted_indices = []
    processed_count = 0
    
    # Starting multiprocessing pool
    with Pool(processes=num_cores) as pool:
        for result in pool.imap_unordered(scan_chunk_worker, tasks):
            if result:
                corrupted_indices.extend(result)
                
            processed_count += chunk_size
            current_progress = min(processed_count, total_ecgs)
            
            # Print progress
            print(f"Scanned {current_progress}/{total_ecgs} | Corrupted found so far: {len(corrupted_indices)}", end='\r')

    elapsed = time.time() - start_time
    
    print("\n\n--- Scan Complete ---")
    print(f"Total time: {elapsed/60:.2f} minutes")
    print(f"Total corrupted ECGs found: {len(corrupted_indices)}")
    
    # Save the bad indices to a CSV file 
    if corrupted_indices:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'corrupted_ecg_indices.csv')
        
        # Sort them numerically before saving
        corrupted_indices.sort()
        df = pd.DataFrame({'corrupted_index': corrupted_indices})
        df.to_csv(save_path, index=False)
        print(f"Saved corrupted index list to:\n  {save_path}")
    else:
        print("Incredible! Your dataset is 100% clean. No missing data found.")

if __name__ == "__main__":
    locate_corrupted_ecgs_parallel(FILE_PATH, LOG_DIR)