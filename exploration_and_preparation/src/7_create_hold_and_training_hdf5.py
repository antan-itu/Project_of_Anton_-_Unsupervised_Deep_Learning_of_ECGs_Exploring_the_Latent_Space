# Script to generate the train and holdout HDF5 files

import h5py
import numpy as np
import time
import os

# --- Configurations ---
BASE_DIR = '/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/data/MIMIC_IV_ECG_HDF5'
INPUT_FILE = os.path.join(BASE_DIR, 'mimic_iv_preprocessed.h5')

TRAIN_FILE = os.path.join(BASE_DIR, 'mimic_iv_train.h5')
HOLDOUT_FILE = os.path.join(BASE_DIR, 'mimic_iv_holdout.h5')

HOLDOUT_SIZE = 150000
CHUNK_SIZE = 50000
RANDOM_SEED = 42

def split_dataset():
    print(f"\n--- Initializing Train/Holdout Split ---")
    print(f"Input: {INPUT_FILE}")
    print(f"Target Holdout Size: {HOLDOUT_SIZE} ECGs")
    
    start_time = time.time()
    
    with h5py.File(INPUT_FILE, 'r') as f_in:
        dset_in = f_in['rhythm_filtered']
        total_ecgs = dset_in.shape[0]
        
        if total_ecgs <= HOLDOUT_SIZE:
            print(f"Error: Total ECGs ({total_ecgs}) is smaller than requested holdout size.")
            return
            
        train_size = total_ecgs - HOLDOUT_SIZE
        print(f"Total clean ECGs available: {total_ecgs}")
        print(f" -> Allocating {train_size} to Training Set")
        print(f" -> Allocating {HOLDOUT_SIZE} to Holdout Set")
        
        # 1. Generate the Random Boolean Mask
        print("\nGenerating randomized split...")
        np.random.seed(RANDOM_SEED)
        
        # Define boolean arrays to indicate which ECGs go to train and holdout
        is_holdout = np.zeros(total_ecgs, dtype=bool)
        holdout_indices = np.random.choice(total_ecgs, HOLDOUT_SIZE, replace=False)
        is_holdout[holdout_indices] = True
        is_train = ~is_holdout
        
        # Preparing the output HDF5 files
        with h5py.File(TRAIN_FILE, 'w') as f_train, h5py.File(HOLDOUT_FILE, 'w') as f_holdout:
            
            dset_train = f_train.create_dataset('rhythm_filtered', 
                                                shape=(train_size, 5000, 8), 
                                                dtype=np.float32, 
                                                chunks=(250, 5000, 8))
                                                
            dset_holdout = f_holdout.create_dataset('rhythm_filtered', 
                                                    shape=(HOLDOUT_SIZE, 5000, 8), 
                                                    dtype=np.float32, 
                                                    chunks=(250, 5000, 8))
            
            train_write_idx = 0
            holdout_write_idx = 0
            
            # Write the ECG data in chunks
            print(f"\n--- Writing Data in Chunks (Size: {CHUNK_SIZE}) ---")
            for i in range(0, total_ecgs, CHUNK_SIZE):
                end_idx = min(i + CHUNK_SIZE, total_ecgs)
                
                # Load chunk and masks into RAM
                chunk = dset_in[i:end_idx]
                chunk_train_mask = is_train[i:end_idx]
                chunk_holdout_mask = is_holdout[i:end_idx]
                
                # Split chunk
                train_data = chunk[chunk_train_mask]
                holdout_data = chunk[chunk_holdout_mask]
                
                # Write to Train File
                if len(train_data) > 0:
                    f_train['rhythm_filtered'][train_write_idx : train_write_idx + len(train_data)] = train_data
                    train_write_idx += len(train_data)
                    
                # Write to Holdout File
                if len(holdout_data) > 0:
                    f_holdout['rhythm_filtered'][holdout_write_idx : holdout_write_idx + len(holdout_data)] = holdout_data
                    holdout_write_idx += len(holdout_data)
                    
                print(f"Processed: {end_idx}/{total_ecgs}", end='\r')
                
            # Adding metadata to both files
            print("\n\n--- Adding Metadata ---")
            gt_in = f_in['GT']
            gt_train = f_train.create_group('GT')
            gt_holdout = f_holdout.create_group('GT')
            
            for key in gt_in.keys():
                print(f" -> Splitting column: {key}")
                original_data = gt_in[key][:]
                
                # Apply the exact same masks to the metadata
                train_meta = original_data[is_train]
                holdout_meta = original_data[is_holdout]
                
                gt_train.create_dataset(key, data=train_meta)
                gt_holdout.create_dataset(key, data=holdout_meta)

    elapsed = time.time() - start_time
    
    print("\n--- Split Complete ---")
    print(f"Total time: {elapsed/60:.2f} minutes")
    print(f"Train File Saved: {TRAIN_FILE}")
    print(f"Holdout File Saved: {HOLDOUT_FILE}")

if __name__ == "__main__":
    split_dataset()