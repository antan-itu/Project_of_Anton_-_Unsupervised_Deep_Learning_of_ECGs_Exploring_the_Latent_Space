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

def check_true_patient_leakage():
    print("Loading record_list.csv for patient mapping...")
    try:
        df_records = pd.read_csv(CSV_PATH)
        
        if 'study_id' not in df_records.columns or 'subject_id' not in df_records.columns:
            print("Error: The CSV file does not contain 'study_id' and 'subject_id' columns.")
            return
            
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
        
        train_patients = [study_to_subject.get(study, f"UNMAPPED_{study}") for study in train_studies]
        holdout_patients = [study_to_subject.get(study, f"UNMAPPED_{study}") for study in holdout_studies]
        
        all_patients = train_patients + holdout_patients
        
        # Patient-Level Math
        train_unique = set(train_patients)
        holdout_unique = set(holdout_patients)
        
        overlap = train_unique.intersection(holdout_unique)
        train_only = train_unique - overlap
        holdout_only = holdout_unique - overlap
        total_unique_patients = len(train_unique.union(holdout_unique))
        
        overlap_count = len(overlap)
        patient_leakage_percentage = (overlap_count / total_unique_patients) * 100 if total_unique_patients > 0 else 0
        
        # ECG-Level Math (The Contamination Count)
        # We count how many individual ECGs belong to the patients in the 'overlap' set.
        # Converting overlap to a set for O(1) lookup speed during the loop.
        overlap_set = set(overlap) 
        
        leaked_ecgs_train = sum(1 for p in train_patients if p in overlap_set)
        leaked_ecgs_holdout = sum(1 for p in holdout_patients if p in overlap_set)
        total_leaked_ecgs = leaked_ecgs_train + leaked_ecgs_holdout
        total_ecgs = len(all_patients)
        ecg_leakage_percentage = (total_leaked_ecgs / total_ecgs) * 100 if total_ecgs > 0 else 0
        
        # ECGs per patient statistics
        id_counts = list(collections.Counter(all_patients).values())
        mean_ecgs = np.mean(id_counts)
        median_ecgs = np.median(id_counts)
        max_ecgs = np.max(id_counts)
        min_ecgs = np.min(id_counts)
        
        print("\n--- True Patient Leakage Report ---")
        print(f"Unique patients across both sets: {total_unique_patients:,}")
        print(f"Unique patients in Train:           {len(train_only):,}")
        print(f"Unique patients in Holdout:         {len(holdout_only):,}")
        print(f"Overlapping Patients:               {overlap_count:,}")
        print(f"Leakage Percentage:                 {patient_leakage_percentage:.2f}%")
        
        print("\n--- Leaked ECGs (Recordings) ---")
        print(f"Total ECGs across both sets:        {total_ecgs:,}")
        print(f"ECGs with leaked patients:          {total_leaked_ecgs:,}")
        print(f"  - ECGs in Train:                  {leaked_ecgs_train:,}")
        print(f"  - ECGs in Holdout:                {leaked_ecgs_holdout:,}")
        print(f"Percentage of all leaked ECGs:      {ecg_leakage_percentage:.2f}%")
        
        print("\n--- ECGs Per Patient Statistics ---")
        print(f"Mean ECGs per patient:   {mean_ecgs:.2f}")
        print(f"Median ECGs per patient: {median_ecgs:.1f}")
        print(f"Min ECGs per patient:    {min_ecgs}")
        print(f"Max ECGs per patient:    {max_ecgs}")
        
        print("\nGenerating Patient Distribution Bar Chart...")
        plt.figure(figsize=(8, 6))
        
        categories = ['Train Only', 'Holdout Only', 'Overlapping (Leakage)']
        counts = [len(train_only), len(holdout_only), overlap_count]
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        
        bars = plt.bar(categories, counts, color=colors, edgecolor='black')
        
        plt.title('Patient Distribution Between Datasets', fontsize=16, fontweight='bold', pad=15)
        plt.ylabel('Number of Unique Patients', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=13)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.gca().set_axisbelow(True)
        
        plt.ylim(0, max(counts) * 1.15)
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + (max(counts)*0.02),
                     f"{yval:,}", ha='center', va='bottom', fontsize=13, fontweight='bold')
                     
        plt.tight_layout()
        plot_path_1 = os.path.join(OUTPUT_DIR, "Patient_Leakage.png")
        plt.savefig(plot_path_1, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Chart saved to: {plot_path_1}")

        print("\nGenerating ECGs per Patient Distribution...")
        plt.figure(figsize=(8, 6))
        
        bins = np.arange(1, max_ecgs + 2) - 0.5 
        
        plt.hist(id_counts, bins=bins, color='#7f7f7f', edgecolor='black')
        
        # Plot Mean and Median lines
        plt.axvline(median_ecgs, color='#1f77b4', linestyle='solid', linewidth=2.5, label=f'Median: {median_ecgs:.1f}')
        
        plt.title('Distribution of ECGs per Patient', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Number of ECGs', fontsize=14)
        plt.ylabel('Number of Patients (Log Scale)', fontsize=14)
        plt.yscale('log') 
        plt.tick_params(axis='both', which='major', labelsize=13)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.gca().set_axisbelow(True)
        
        plt.xlim(0.5, max_ecgs + 0.5)
        
        # Add legend for the mean and median lines
        plt.legend(loc='upper right', fontsize=13, framealpha=1.0)
        
        plt.tight_layout()
        plot_path_2 = os.path.join(OUTPUT_DIR, "ECG_Per_Patient_Distribution.png")
        plt.savefig(plot_path_2, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Chart saved to: {plot_path_2}")

if __name__ == "__main__":
    check_true_patient_leakage()