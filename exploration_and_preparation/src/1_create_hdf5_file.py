# This script reads the ECG CSV files, and saves them into a HDF5 file with the corresponding metadata.

import os
import pandas as pd
import numpy as np
import h5py
import time
from multiprocessing import Pool, cpu_count

# Defining paths
BASE_DIR = '/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/data/unzipped/MIMIC_IV_ECG_CSV_MICROVOLTS_v3'
OUTPUT_DIR = '/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/data/MIMIC_IV_ECG_HDF5'

# ECG leads: I, II, V1, V2, V3, V4, V5, V6
INDEPENDENT_LEAD_INDICES = [0, 1, 6, 7, 8, 9, 10, 11]

def read_csv_worker(row_dict):
    """
    The function runsparallel across multiple CPU cores.
    """
    file_id = str(row_dict['path']).split('/')[-1]
    csv_path = os.path.join(BASE_DIR, 'files', f"{file_id}.csv")
    
    if not os.path.exists(csv_path):
        return None
        
    try:
        # Load the CSV and extract the 8 leads
        signal_df = pd.read_csv(csv_path, header=None)
        signal_matrix = signal_df.iloc[:, INDEPENDENT_LEAD_INDICES].values
        
        if signal_matrix.shape == (5000, 8):
            # Return the matrix and metadata
            return {
                'matrix': signal_matrix.astype(np.float32),
                'metadata': row_dict
            }
    except Exception:
        pass
        
    return None

def build_full_dataset_parallel(base_dir, save_dir, chunk_size=100000):
    print("\n--- Initializing Dataset Generation ---")
    start_time = time.time()
    
    record_list_path = os.path.join(base_dir, 'record_list.csv')
    ground_truth_path = os.path.join(base_dir, 'ground_truth_h5.csv')
    
    print(" -> Loading CSVs...")
    records_df = pd.read_csv(record_list_path)
    gt_df = pd.read_csv(ground_truth_path, sep=None, engine='python', on_bad_lines='skip')
    merged_df = pd.merge(records_df, gt_df, on='study_id', how='inner')
    
    merged_df['report_0'] = merged_df['report_0'].fillna('')
    
    # Convert dataframe to a list of dictionaries for multiprocessing
    tasks = merged_df.to_dict('records')
    total_files = len(tasks)
    print(f" -> Total target files identified: {total_files}")
    
    os.makedirs(save_dir, exist_ok=True)
    h5_output_path = os.path.join(save_dir, 'mimic_iv_aligned_full.h5')
    
    # Determine number of CPU cores
    num_cores = max(1, cpu_count() - 2)
    print(f" -> Using {num_cores} cores...")
    
    valid_metadata = []
    valid_count = 0
    buffer_signals = []
    
    with h5py.File(h5_output_path, 'w') as f:
        # Preparing HDF5 space
        dset = f.create_dataset('rhythm_filtered', 
                                shape=(total_files, 5000, 8), 
                                maxshape=(None, 5000, 8),
                                dtype=np.float32, 
                                chunks=(250, 5000, 8))
        
        print(f"\n--- Beginning Extraction (Writing in chunks of {chunk_size}) ---")
        
        # Initializing multiprocessing pool
        with Pool(processes=num_cores) as pool:
            # imap_unordered is the fastest method. It yields results as soon as any worker finishes.
            for result in pool.imap_unordered(read_csv_worker, tasks, chunksize=500):
                if result is not None:
                    buffer_signals.append(result['matrix'])
                    valid_metadata.append(result['metadata'])
                    valid_count += 1
                
                # Write to HDF5 in chunks
                if len(buffer_signals) >= chunk_size:
                    start_idx = valid_count - chunk_size
                    end_idx = valid_count
                    dset[start_idx:end_idx] = np.array(buffer_signals, dtype=np.float32)
                    buffer_signals.clear()
                    
                    print(f"Progress - files stored: {valid_count} / {total_files}")
                    
        # Save any remaining ECGs
        if buffer_signals:
            start_idx = valid_count - len(buffer_signals)
            end_idx = valid_count
            dset[start_idx:end_idx] = np.array(buffer_signals, dtype=np.float32)
            buffer_signals.clear()

        print("\n--- Finalizing Dataset ---")
        print(f"Changing HDF5 file size to {valid_count} files.")
        dset.resize(valid_count, axis=0)

        print("Adding Ground Truth Metadata...")
        final_metadata_df = pd.DataFrame(valid_metadata)
        
        gt_group = f.create_group('GT')
        gt_group.create_dataset('study_id', data=final_metadata_df['study_id'].values)
        
        # Save report columns dynamically
        report_columns = [col for col in final_metadata_df.columns if col.startswith('report_')]
        for col_name in report_columns:
            clean_text_array = final_metadata_df[col_name].fillna('').astype(str).values.astype('S')
            gt_group.create_dataset(col_name, data=clean_text_array)

    elapsed = time.time() - start_time
    print(f"\n--- Pipeline Complete ---")
    print(f"Total time: {elapsed/60:.2f} minutes")
    print(f"Files aligned and saved to:\n  {h5_output_path}")

if __name__ == "__main__":
    build_full_dataset_parallel(base_dir=BASE_DIR, save_dir=OUTPUT_DIR, chunk_size=100000)