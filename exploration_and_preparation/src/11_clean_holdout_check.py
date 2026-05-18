import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import os

# --- Configurations ---
BASE_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space"

TRAIN_PATH = f"{BASE_DIR}/data/MIMIC_IV_ECG_HDF5/mimic_iv_train.h5"
HOLDOUT_PATH = f"{BASE_DIR}/data/MIMIC_IV_ECG_HDF5/mimic_iv_holdout.h5"
CSV_PATH = f"{BASE_DIR}/data/unzipped/MIMIC_IV_ECG_CSV_MICROVOLTS_v3/record_list.csv"
OUTPUT_DIR = f"{BASE_DIR}/exploration_and_preparation/logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_clean_holdout_distribution():
    print("Loading record_list.csv for patient mapping...")
    try:
        df_records = pd.read_csv(CSV_PATH)
        study_to_subject = dict(zip(df_records['study_id'].astype(str), df_records['subject_id'].astype(str)))
        print(f"Successfully loaded {len(study_to_subject)} study-to-patient mappings.")
    except FileNotFoundError:
        print(f"Error: Could not locate CSV at {CSV_PATH}")
        return

    print("\nOpening HDF5 files to extract study IDs...")
    with h5py.File(TRAIN_PATH, 'r') as f_train, h5py.File(HOLDOUT_PATH, 'r') as f_holdout:
        
        def get_study_ids(hdf5_file):
            raw_data = hdf5_file['GT']['study_id'][:]
            return [val.decode('utf-8') if isinstance(val, bytes) else str(val) for val in raw_data]
            
        train_studies = get_study_ids(f_train)
        holdout_studies = get_study_ids(f_holdout)
        
        # Map studies to patients
        train_patients = set([study_to_subject.get(study, f"UNMAPPED_{study}") for study in train_studies])
        holdout_patients = [study_to_subject.get(study, f"UNMAPPED_{study}") for study in holdout_studies]

    # --- Isolate Clean Holdout ---
    print("\nFiltering for clean holdout patients (no leakage from train)...")
    # Keep only the patient IDs in the holdout set that DO NOT appear in the train set
    clean_holdout_ecgs = [p for p in holdout_patients if p not in train_patients]
    
    # Count how many ECGs belong to each clean patient
    id_counts = list(collections.Counter(clean_holdout_ecgs).values())
    
    if not id_counts:
        print("Error: No clean holdout patients found.")
        return

    # Statistics
    mean_ecgs = np.mean(id_counts)
    median_ecgs = np.median(id_counts)
    max_ecgs = np.max(id_counts)
    min_ecgs = np.min(id_counts)
    
    print("\n--- Clean Holdout Statistics ---")
    print(f"Total Clean Holdout Patients: {len(set(clean_holdout_ecgs)):,}")
    print(f"Total Clean Holdout ECGs:     {len(clean_holdout_ecgs):,}")
    print(f"Mean ECGs per patient:        {mean_ecgs:.2f}")
    print(f"Median ECGs per patient:      {median_ecgs:.1f}")
    print(f"Min ECGs per patient:         {min_ecgs}")
    print(f"Max ECGs per patient:         {max_ecgs}")

    # --- Plotting ---
    print("\nGenerating Clean Holdout ECGs per Patient Distribution...")
    plt.figure(figsize=(8, 6))
    
    bins = np.arange(1, max_ecgs + 2) - 0.5 
    
    plt.hist(id_counts, bins=bins, color='#2ca02c', edgecolor='black') # Using green to denote 'clean'
    
    # Plot Mean and Median lines
    plt.axvline(median_ecgs, color='#1f77b4', linestyle='solid', linewidth=2.5, label=f'Median: {median_ecgs:.1f}')
    
    plt.title('Distribution of ECGs per Patient (Clean Holdout Subset)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Number of ECGs', fontsize=14)
    plt.ylabel('Number of Patients (Log Scale)', fontsize=14)
    plt.yscale('log') 
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.gca().set_axisbelow(True)
    
    plt.xlim(0.5, max_ecgs + 0.5)
    
    # Add legend
    plt.legend(loc='upper right', fontsize=13, framealpha=1.0)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "Clean_Holdout_ECG_Per_Patient_Distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Chart saved to: {plot_path}")

if __name__ == "__main__":
    plot_clean_holdout_distribution()