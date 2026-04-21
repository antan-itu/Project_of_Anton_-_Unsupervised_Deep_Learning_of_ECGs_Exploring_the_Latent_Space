import pandas as pd
import re
import os

# --- Define paths ---
BASE_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/Exploration and Preparation/logs"
OUTPUT_CSV_PRIMARY = os.path.join(BASE_DIR, "primary_labels.csv")
OUTPUT_CSV_SECONDARY = os.path.join(BASE_DIR, "secondary_labels.csv")
FILE_PATH = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/Data/Full training dataset/training_dataset.h5"

print("Extracting Pandas DataFrame from 'GT'...\n")
df_ground_truth = pd.read_hdf(FILE_PATH, key='GT')

total_dataset_size = len(df_ground_truth)
report_cols = [f'report_{i}' for i in range(18)]

# 1. STRICT PRIMARY MATCHES
EXACT_TARGETS = [
    "ATRIAL FIBRILLATION",
    "Atrial fibrillation",
    "Atrial fibrillation."
]

# 2. BROAD SECONDARY MATCH
secondary_regex = r'(?i)atrial fibrillation'

primary_mask = pd.Series(False, index=df_ground_truth.index)
secondary_mask = pd.Series(False, index=df_ground_truth.index)

for col in report_cols:
    if col in df_ground_truth.columns:
        # Clean up whitespace immediately
        col_cleaned = df_ground_truth[col].fillna('').astype(str).str.strip()
        
        # Primary: Only flag True if the cell matches our exact list perfectly
        primary_mask |= col_cleaned.isin(EXACT_TARGETS)
        
        # Secondary: Broad case-insensitive search
        secondary_mask |= col_cleaned.str.contains(secondary_regex, regex=True)

# Count the hits
primary_hits = primary_mask.sum()
# Only count secondary hits if they didn't already trigger the primary match
additional_hits = (secondary_mask & ~primary_mask).sum() 
total_hits = primary_hits + additional_hits

# Extract distinct original labels into two separate sets
unique_labels_primary = set()
unique_labels_secondary = set()

for col in report_cols:
    if col in df_ground_truth.columns:
        valid_cells = df_ground_truth[col].dropna().astype(str)
        for cell_text in valid_cells:
            text_clean = cell_text.strip()
            if not text_clean:
                continue
            
            if text_clean in EXACT_TARGETS:
                unique_labels_primary.add(text_clean)

            elif re.search(secondary_regex, text_clean):
                unique_labels_secondary.add(text_clean)

# Sort the unique labels by length (shortest to longest)
sorted_primary = sorted(list(unique_labels_primary), key=len)
sorted_secondary = sorted(list(unique_labels_secondary), key=len)

# Convert to DataFrames and save to CSV
pd.DataFrame({'Original_Label_Primary': sorted_primary}).to_csv(OUTPUT_CSV_PRIMARY, index=False)
pd.DataFrame({'Original_Label_Secondary': sorted_secondary}).to_csv(OUTPUT_CSV_SECONDARY, index=False)

# Print the requested statistics
print(f"--- Hit Statistics (Row/ECG Level) ---")
print(f"Total ECGs in dataset: {total_dataset_size}")
print(f"ECGs matching STRICT exact targets: {primary_hits}")
print(f"ECGs matching BROAD variations (Excluded): {additional_hits}")
print(f"Total Combined ECGs with AFib: {total_hits}")
print(f"--------------------------------------\n")

print(f"Success! Files saved to:")
print(f" 1. {OUTPUT_CSV_PRIMARY} ({len(sorted_primary)} distinct exact labels)")
print(f" 2. {OUTPUT_CSV_SECONDARY} ({len(sorted_secondary)} distinct excluded labels)")