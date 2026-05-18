import os
import pandas as pd
import h5py

# --- 1. SET DIRECTORIES ---
BASE_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space"
DATA_PATH = os.path.join(BASE_DIR, "data/MIMIC_IV_ECG_HDF5/mimic_iv_holdout.h5")
REVIEW_DIR = os.path.join(BASE_DIR, "exploration_and_preparation/logs/manual_review_400_ecgs")

# Input and Output CSV paths
INPUT_CSV = os.path.join(REVIEW_DIR, "manual_afib_review_csv.csv")
OUTPUT_CSV = os.path.join(REVIEW_DIR, "ground_truth_afib_review.csv")

# --- 2. HELPER FUNCTION ---
EXACT_TARGETS = ["ATRIAL FIBRILLATION", "Atrial fibrillation", "Atrial fibrillation."]

def extract_afib_labels(h5_file_path):
    df_gt_dict = {}
    with h5py.File(h5_file_path, 'r') as f:
        gt_group = f['GT']
        report_cols = [key for key in gt_group.keys() if key.startswith('report_')]
        for col in report_cols:
            df_gt_dict[col] = [val.decode('utf-8') for val in gt_group[col][:]]

    df_gt = pd.DataFrame(df_gt_dict)
    mask = pd.Series(False, index=df_gt.index)
    for col in report_cols:
        if col in df_gt.columns:
            mask |= df_gt[col].fillna('').astype(str).str.strip().isin(EXACT_TARGETS)
    return mask.astype(int).values

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"Loading manual review indices from:\n{INPUT_CSV}")
    review_df = pd.read_csv(INPUT_CSV)
    
    print(f"\nExtracting all ground truth labels from HDF5:\n{DATA_PATH}")
    all_labels = extract_afib_labels(DATA_PATH)
    
    print("\nMapping actual labels to the 400 selected indices...")
    # The index in the `all_labels` array corresponds directly to the HDF5 index
    review_df['Actual_AFib_Label'] = review_df['HDF5_Index'].apply(lambda x: all_labels[x])
    
    print(f"\nSaving updated CSV with ground truth to:\n{OUTPUT_CSV}")
    review_df.to_csv(OUTPUT_CSV, index=False)
    
    print("\nProcess complete. First 5 rows of the new CSV:")
    print(review_df.head())