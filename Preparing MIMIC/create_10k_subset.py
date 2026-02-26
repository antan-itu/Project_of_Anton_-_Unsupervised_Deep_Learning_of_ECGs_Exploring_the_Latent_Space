import h5py
import numpy as np
import os

# --- Paths ---
SOURCE_PATH = "/home/akokholm/mnt/SUN-BMI-EC-MIMIC-ECG/ECG.hdf5"
TARGET_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/Data/MIMIC-IV_Subset"
TARGET_PATH = os.path.join(TARGET_DIR, "ecg_dataset_10k_8lead.h5")

os.makedirs(TARGET_DIR, exist_ok=True)

def extract_subset():
    print(f"Opening massive source file: {SOURCE_PATH}")
    
    with h5py.File(SOURCE_PATH, 'r') as source_h5:
        # Targeting the filtered ECG dataset
        dataset_key = 'rhythm_filtered' 
        print(f"Accessing Dataset: '{dataset_key}'")
        
        # Slicing the first 10,000 samples. 
        # Source shape is (800035, 5000, 8)
        print("Extracting the first 10,000 samples into RAM...")
        raw_data = source_h5[dataset_key][:10000]
        
        # Convert from int32 to float32 for PyTorch compatibility
        raw_data = raw_data.astype(np.float32)
        print(f"Raw shape extracted: {raw_data.shape} | Type: {raw_data.dtype}")
        
        # Permute from TensorFlow format (Batch, Length, Channels) 
        # To PyTorch format (Batch, Channels, Length) -> (10000, 8, 5000)
        subset_data = np.transpose(raw_data, (0, 2, 1))

        print(f"Final PyTorch-ready shape: {subset_data.shape}")
        
    print(f"\nWriting to new HDF5 file: {TARGET_PATH}")
    with h5py.File(TARGET_PATH, 'w') as target_h5:
        target_h5.create_dataset("ecg_data", data=subset_data, dtype=np.float32)
        
    print("Success! Subset created and ready for training.")

if __name__ == "__main__":
    extract_subset()