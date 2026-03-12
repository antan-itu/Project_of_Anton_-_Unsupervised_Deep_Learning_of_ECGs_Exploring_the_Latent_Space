import pandas as pd
import scipy.stats as st
import numpy as np
import os

# Define paths
INPUT_CSV = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/Exploration and Preparation/logs/results.CSV"
OUTPUT_CSV = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/Exploration and Preparation/logs/results_with_ci.CSV"

print(f"Loading existing results from:\n{INPUT_CSV}\n")

# FIXED: Added sep=';' to handle semicolon-separated files correctly
df = pd.read_csv(INPUT_CSV, sep=';')

def calculate_95_ci(row):
    try:
        n = int(row['k_folds'])
        mean = float(row['Avg_Val_RMSE'])
        std = float(row['Std_Val_RMSE'])
        
        if n <= 1 or pd.isna(mean) or pd.isna(std):
            return "N/A"
            
        std_err = std / np.sqrt(n)
        margin_of_error = st.t.ppf(1 - 0.025, n - 1) * std_err
        
        ci_lower = mean - margin_of_error
        ci_upper = mean + margin_of_error
        
        return f"[{ci_lower:.3f}, {ci_upper:.3f}]"
    except (KeyError, ValueError):
        return "N/A"

print("Calculating 95% Confidence Intervals for Val_RMSE...")

# Apply the function row by row
df['CI_Val_RMSE'] = df.apply(calculate_95_ci, axis=1)

# Reorder the columns to place CI_Val_RMSE exactly after Std_Val_RMSE
cols = list(df.columns)
if 'CI_Val_RMSE' in cols:
    cols.remove('CI_Val_RMSE')

if 'Std_Val_RMSE' in cols:
    insert_loc = cols.index('Std_Val_RMSE') + 1
    cols.insert(insert_loc, 'CI_Val_RMSE')
    df = df[cols]
else:
    print("Warning: 'Std_Val_RMSE' column not found. The CI column was appended at the end.")

# FIXED: Added sep=';' here as well so your output matches your input format
df.to_csv(OUTPUT_CSV, sep=';', index=False)

print(f"\nSuccess! Processed {len(df)} rows.")
print(f"Updated results have been saved to:\n{OUTPUT_CSV}")