import pandas as pd
import re
import os

# Define paths (Updated to save two separate files)
BASE_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/Exploration and Preparation/logs"
OUTPUT_CSV_PRIMARY = os.path.join(BASE_DIR, "afib_primary_labels.csv")
OUTPUT_CSV_SECONDARY = os.path.join(BASE_DIR, "afib_secondary_labels.csv")
FILE_PATH = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/Data/Full training dataset/training_dataset.h5"

print("Extracting Pandas DataFrame from 'GT'...\n")
df_ground_truth = pd.read_hdf(FILE_PATH, key='GT')

total_dataset_size = len(df_ground_truth)
report_cols = [f'report_{i}' for i in range(18)]

# Regex patterns (Added ?: to the secondary regex to prevent Pandas UserWarning)
primary_regex = r'(?i)\batrial\s+fibrillation\b'
secondary_regex = r'(?i)\b(?:afib|a-fib|a\.fib|a\.\s*fib|a\s+fib|af|a\.f\.|atrial\s+fib|atrial\s+fibrilation|fibrillation,\s*atrial)\b'

primary_pattern = re.compile(primary_regex)
secondary_pattern = re.compile(secondary_regex)

print(f"Calculating statistics across {total_dataset_size} total ECGs...\n")

# Combine all report columns into a single string per row to count ECGs accurately
combined_reports = df_ground_truth[report_cols].fillna('').astype(str).agg(' '.join, axis=1)

# Calculate row-level hits using pandas vectorization
primary_mask = combined_reports.str.contains(primary_regex, regex=True)
secondary_mask = combined_reports.str.contains(secondary_regex, regex=True)

# Count the hits
primary_hits = primary_mask.sum()
# Only count secondary hits if they didn't already trigger the primary match
additional_hits = (secondary_mask & ~primary_mask).sum() 
total_afib_hits = primary_hits + additional_hits

# Extract distinct original labels into two separate sets
unique_labels_primary = set()
unique_labels_secondary = set()

for col in report_cols:
    if col in df_ground_truth.columns:
        valid_cells = df_ground_truth[col].dropna().astype(str)
        for cell_text in valid_cells:
            text_clean = cell_text.strip()
            
            # If it matches the primary pattern, it goes to the primary set
            if primary_pattern.search(text_clean):
                unique_labels_primary.add(text_clean)
            # If it DOES NOT match the primary, but matches the secondary, it goes to the secondary set
            elif secondary_pattern.search(text_clean):
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
print(f"ECGs matching 'Atrial fibrillation': {primary_hits}")
print(f"Additional ECGs from other patterns (AFib, AF, etc.): {additional_hits}")
print(f"Total Combined AFib ECGs: {total_afib_hits}")
print(f"--------------------------------------\n")

print(f"Success! Files saved to:")
print(f" 1. {OUTPUT_CSV_PRIMARY} ({len(sorted_primary)} distinct labels)")
print(f" 2. {OUTPUT_CSV_SECONDARY} ({len(sorted_secondary)} distinct labels)")