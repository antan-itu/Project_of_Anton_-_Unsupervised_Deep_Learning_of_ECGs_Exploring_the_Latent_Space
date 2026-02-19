import os
import sys
import subprocess
import numpy as np
import tensorflow as tf

# ==========================================
# 0. Auto-Install Dependencies
# ==========================================
def ensure_dependencies():
    """Automatically installs required packages if missing."""
    print("Checking dependencies (h5py, tqdm)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "h5py", "tqdm", "--quiet"])
    print("Dependencies verified! Starting script...\n")

ensure_dependencies()

import h5py
from tqdm import tqdm

# ==========================================
# 1. Configuration & Paths
# ==========================================
HDF5_PATH = "/home/akokholm/mnt/SUN-BMI-EC-MIMIC-ECG/ECG.hdf5"
OUTPUT_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/Data/MIMIC-IV"

# Leveraging your 377 GB of RAM and Threadripper CPU
SAMPLES_PER_FILE = 50000  

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. Serialization Function
# ==========================================
def serialize_example(ecg_array):
    """Converts a single ECG numpy array to a raw byte string for TFRecords."""
    ecg_array = ecg_array.astype(np.float32)
    ecg_bytes = ecg_array.tobytes()
    
    feature = {
        'ecg': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ecg_bytes]))
    }
    
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# ==========================================
# 3. Main Conversion Loop
# ==========================================
def convert_hdf5_to_tfrecords():
    print(f"Opening HDF5 file: {HDF5_PATH}\n")
    
    with h5py.File(HDF5_PATH, 'r') as h5f:
        # ---------------------------------------------------------
        # HARDCODED TARGET DATASET
        # You can change this to 'rhythm_filtered' if you prefer!
        # ---------------------------------------------------------
        target_name = 'rhythm'
        print(f"Targeting specific dataset: '{target_name}'")
        
        data = h5f[target_name]
        total_samples = data.shape[0]
        
        num_files = int(np.ceil(total_samples / SAMPLES_PER_FILE))
        print(f"Converting {total_samples} ECGs into {num_files} TFRecord chunks...\n")
        
        for file_idx in range(num_files):
            start_idx = file_idx * SAMPLES_PER_FILE
            end_idx = min((file_idx + 1) * SAMPLES_PER_FILE, total_samples)
            
            record_file = os.path.join(OUTPUT_DIR, f"mimic_chunk_{file_idx+1:03d}.tfrecord")
            print(f"Writing {record_file} (Samples {start_idx} to {end_idx-1})...")
            
            # Read chunk into RAM
            chunk_data = data[start_idx:end_idx]
            
            with tf.io.TFRecordWriter(record_file) as writer:
                # Progress bar for the current chunk
                for i in tqdm(range(chunk_data.shape[0]), desc=f"Chunk {file_idx+1}/{num_files}", leave=False):
                    tf_example = serialize_example(chunk_data[i])
                    writer.write(tf_example)
                    
    print("\n🎉 Conversion Complete! All TFRecords are saved and ready for the RTX 4090.")

if __name__ == "__main__":
    convert_hdf5_to_tfrecords()