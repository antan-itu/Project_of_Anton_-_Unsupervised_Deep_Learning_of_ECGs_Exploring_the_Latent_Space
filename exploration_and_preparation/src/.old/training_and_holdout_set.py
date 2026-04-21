import os
import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def write_subset_h5(target_path, source_h5_path, subset_df, desc_prefix):
    """
    Writes the filtered GT table and streams the corresponding rhythm arrays into a new HDF5.
    """
    print(f"\n--- Saving {desc_prefix} Dataset to {target_path} ---")
    
    # 1. Save the GT table
    print("Writing GT metadata table...")
    clean_df = subset_df.drop(columns=['match_id'], errors='ignore')
    clean_df.to_hdf(target_path, key='GT', format='table', mode='w')

    # 2. Sort indices to enable fast HDF5 disk streaming
    sorted_df = clean_df.sort_index()
    indices_to_extract = sorted_df.index.to_numpy()

    # 3. Stream the massive rhythm arrays
    batch_size = 1000
    
    with h5py.File(source_h5_path, 'r') as src, h5py.File(target_path, 'a') as dst:
        for ds_name in ['rhythm', 'rhythm_filtered']:
            if ds_name not in src:
                continue
                
            src_ds = src[ds_name]
            print(f"Preparing array shape for {ds_name}...")
            
            # Target shape: (Number of subset rows, timepoints, leads)
            target_shape = (len(indices_to_extract),) + src_ds.shape[1:]
            
            # Create empty dataset in the new file
            dst_ds = dst.create_dataset(
                ds_name, 
                shape=target_shape, 
                dtype=src_ds.dtype, 
                chunks=True # Enables efficient disk writing
            )

            # Copy data in batches
            for i in tqdm(range(0, len(indices_to_extract), batch_size), desc=f"Streaming {ds_name}"):
                batch_idx = indices_to_extract[i : i + batch_size]
                dst_ds[i : i + len(batch_idx)] = src_ds[batch_idx]

def main():
    # --- 1. CONFIGURATION ---
    base_dir = r"/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space"
    
    source_h5_path = r"/home/akokholm/mnt/SUN-BMI-EC-MIMIC-ECG/ECG.hdf5" 
    log_path = os.path.join(base_dir, "Preparing MIMIC/ecg_filled_log.csv")
    
    train_dir = os.path.join(base_dir, "Data/Full training dataset")
    holdout_dir = os.path.join(base_dir, "Data/Holdout set")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(holdout_dir, exist_ok=True)
    
    train_h5_path = os.path.join(train_dir, "training_dataset.h5")
    holdout_h5_path = os.path.join(holdout_dir, "holdout_dataset.h5")

    # --- 2. LOAD BAD FILES ---
    print("Loading exclusion log...")
    log_df = pd.read_csv(log_path)
    bad_files_df = log_df[log_df['Has Empty Cells (And Was Filled)'].str.strip().str.upper() == 'YES']
    
    # Extract the ID without the '.csv' extension
    bad_ids = set(bad_files_df['File Name'].str.replace('.csv', '', regex=False))
    print(f"Identified {len(bad_ids)} bad ECGs to exclude.")

    # --- 3. LOAD & FILTER GT TABLE ---
    print("Loading GT table from HDF5 (this might take a moment)...")
    gt_df = pd.read_hdf(source_h5_path, key='GT')
    print(f"Total ECGs before filtering: {len(gt_df)}")

    # We assume 'study_id' matches the file name.
    gt_df['match_id'] = gt_df['study_id'].astype(str)
    
    # Keep only rows where the match_id is NOT in our list of bad IDs
    valid_df = gt_df[~gt_df['match_id'].isin(bad_ids)]
    print(f"Total ECGs after filtering out empty/filled cells: {len(valid_df)}")

    # --- 4. RANDOM SPLIT ---
    print("\nPerforming random split (Holdout: 150,000)...")
    train_df, holdout_df = train_test_split(
        valid_df, 
        test_size=150000, 
        random_state=42 
    )
    
    print(f"Final Train Set Size: {len(train_df)}")
    print(f"Final Holdout Set Size: {len(holdout_df)}")

    # --- 5. WRITE NEW HDF5 FILES ---
    write_subset_h5(holdout_h5_path, source_h5_path, holdout_df, "Holdout")
    write_subset_h5(train_h5_path, source_h5_path, train_df, "Train")

    print("\nSuccess! Both datasets have been fully generated and saved.")

if __name__ == "__main__":
    main()