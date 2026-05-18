# The script removes ECGs with dead leads.

import h5py
import numpy as np
import time
import os

# --- Configurations ---
BASE_DIR = '/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/data/MIMIC_IV_ECG_HDF5'
INPUT_FILE = os.path.join(BASE_DIR, 'mimic_iv_aligned_full.h5')
OUTPUT_FILE = os.path.join(BASE_DIR, 'mimic_iv_preprocessed.h5')

MAX_DEAD_LEADS_ALLOWED = 0 
CHUNK_SIZE = 50000

def preprocess_and_clean_dataset():
    print(f"\n--- Initializing Dataset Preprocessing ---")
    print(f"Input: {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Dropping ECGs with > {MAX_DEAD_LEADS_ALLOWED} dead leads.\n")
    
    start_time = time.time()
    
    # We will build a global boolean mask to filter the metadata at the end
    global_valid_mask = []
    total_processed = 0
    total_valid = 0
    
    with h5py.File(INPUT_FILE, 'r') as f_in, h5py.File(OUTPUT_FILE, 'w') as f_out:
        
        # Stteing up the input and output
        dset_in = f_in['rhythm_filtered']
        total_ecgs = dset_in.shape[0]
        
        print(f"Total ECGs to process: {total_ecgs}")
        
        # Preparing the output dataset
        dset_out = f_out.create_dataset('rhythm_filtered', 
                                        shape=(total_ecgs, 5000, 8), 
                                        maxshape=(None, 5000, 8),
                                        dtype=np.float32, 
                                        chunks=(250, 5000, 8))
        
        # Filter, and Write the Matrices
        print(f"\n--- Filtering Matrices (Chunk Size: {CHUNK_SIZE}) ---")
        for i in range(0, total_ecgs, CHUNK_SIZE):
            end_idx = min(i + CHUNK_SIZE, total_ecgs)
            
            # Read chunk into RAM
            chunk = dset_in[i:end_idx]
            
            # Calculate flatlines: Peak-to-Peak == 0.0 along the time axis
            dead_leads_per_ecg = (np.ptp(chunk, axis=1) == 0.0).sum(axis=1)
            
            # Create boolean mask for valid ECGs
            valid_mask = dead_leads_per_ecg <= MAX_DEAD_LEADS_ALLOWED
            global_valid_mask.extend(valid_mask.tolist())
            
            # Filter the chunk
            clean_chunk = chunk[valid_mask]
            
            # Write data to the output file
            if len(clean_chunk) > 0:
                start_write = total_valid
                end_write = total_valid + len(clean_chunk)
                dset_out[start_write:end_write] = clean_chunk
                total_valid += len(clean_chunk)
                
            total_processed += (end_idx - i)
            print(f"Processed: {total_processed}/{total_ecgs} | Valid Kept: {total_valid}", end='\r')
            
        # Trimming output dataset
        print(f"\n\n--- Trimming Array ---")
        print(f"Shrinking tensor from {total_ecgs} to {total_valid}...")
        dset_out.resize(total_valid, axis=0)
        
        # Adding metadata 
        print("\n--- Adding Ground Truth Metadata ---")
        global_valid_mask = np.array(global_valid_mask, dtype=bool)
        
        gt_in = f_in['GT']
        gt_out = f_out.create_group('GT')
        
        # Loop through every label column, apply the boolean mask, and save
        for key in gt_in.keys():
            print(f" -> Filtering and copying column: {key}")
            original_data = gt_in[key][:]
            clean_data = original_data[global_valid_mask]
            gt_out.create_dataset(key, data=clean_data)
            
    elapsed = time.time() - start_time
    dropped_count = total_ecgs - total_valid
    
    print("\n--- Preprocessing Complete ---")
    print(f"Total time: {elapsed/60:.2f} minutes")
    print(f"Total ECGs dropped: {dropped_count}")
    print(f"Final Shape: ({total_valid}, 5000, 8)")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    preprocess_and_clean_dataset()