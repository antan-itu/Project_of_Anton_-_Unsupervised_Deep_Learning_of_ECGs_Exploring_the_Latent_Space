import os
import glob
import h5py
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# ==========================================
# Configuration
# ==========================================
# TFRecord path
TFRECORD_PATH = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/" \
                "Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/" \
                "Data/MIMIC-IV_Subset/*.tfrecord"

# Where to save the new HDF5 file
OUTPUT_H5_FILE = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/" \
                 "Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/" \
                 "Data/MIMIC-IV_Subset/ecg_dataset.h5"

EXPECTED_SHAPE = (5000, 12)
TOTAL_SAMPLES = 5087

feature_description = {
    "ecg": tf.io.FixedLenFeature([], tf.string)
}

def parse_tfrecord(example_proto):
    """Parses the raw bytes back into a float32 tensor."""
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    ecg = tf.io.decode_raw(parsed["ecg"], out_type=tf.float32)
    ecg = tf.reshape(ecg, EXPECTED_SHAPE)
    return ecg

def convert_to_h5():
    files = glob.glob(TFRECORD_PATH)
    if not files:
        print(f"Error: No TFRecord files found at {TFRECORD_PATH}")
        return

    print(f"Found {len(files)} TFRecord files. Starting conversion...")
    
    # Create a raw TFRecordDataset
    raw_dataset = tf.data.TFRecordDataset(files)
    
    # Open HDF5 file in write mode
    with h5py.File(OUTPUT_H5_FILE, 'w') as h5f:
        # Create a chunked dataset. 
        # Note the shape is (TOTAL_SAMPLES, 12, 5000) for PyTorch!
        dset = h5f.create_dataset(
            "ecg_data", 
            shape=(TOTAL_SAMPLES, 12, 5000), 
            dtype=np.float32,
            chunks=True # Allows efficient partial reading
        )
        
        # Iterate through the TFRecords and write to HDF5
        for i, raw_record in enumerate(tqdm(raw_dataset, total=TOTAL_SAMPLES, desc="Converting")):
            if i >= TOTAL_SAMPLES:
                break
                
            # Parse the TensorFlow tensor
            ecg_tensor = parse_tfrecord(raw_record)
            
            # Convert to numpy array
            ecg_np = ecg_tensor.numpy()
            
            # Permute to PyTorch format: (5000, 12) -> (12, 5000)
            ecg_np = np.transpose(ecg_np, (1, 0))
            
            # Write directly to the HDF5 dataset
            dset[i] = ecg_np
            
    print(f"\nSuccess! Dataset saved to {OUTPUT_H5_FILE}")
    print(f"Final HDF5 shape: {dset.shape}")

if __name__ == "__main__":
    convert_to_h5()