import os
import json
import numpy as np
import scipy.stats as stats

# --- 1. Configurations ---
EXPERIMENTS_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/model_development/experiments"

### Place "Rank 1 / Best" model FIRST in this list. ###
RUN_NAMES = [
    "GridRun_001_2404_1510",      # Rank 1 (Baseline)
    "GridRun_001_2404_1047",      # Rank 2
    "GridRun_001_2404_1817",      # Rank 3
    "GridRun_002_2404_1945",      # Rank 4
    "GridRun_003_2404_2130"       # Rank 5
]

# Select metric: "XGB_PRAUC", "XGB_AUC", "Val_MSE", "Val_RMSE"
TARGET_METRIC = "XGB_PRAUC"

# --- 2. Helper Function: Holm-Bonferroni ---
def holm_correction(p_values, alpha=0.05):
    """Implementation of Holm-Bonferroni correction."""
    p_values = np.array(p_values)
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    
    p_values_corrected = np.zeros(n)
    
    for i, idx in enumerate(sorted_indices):
        # Calculate corrected p-value
        adj_p = p_values[idx] * (n - i)
        
        # Ensure p-values are monotonically increasing and capped at 1.0
        if i == 0:
            p_values_corrected[idx] = min(1.0, adj_p)
        else:
            p_values_corrected[idx] = min(1.0, max(adj_p, p_values_corrected[sorted_indices[i-1]]))
            
    # Rejection logic is strictly based on whether the corrected p-value is <= alpha
    reject = p_values_corrected <= alpha
            
    return reject, p_values_corrected

# --- 3. Data Extraction ---
models = {}
print(f"Loading '{TARGET_METRIC}' for {len(RUN_NAMES)} models...")

for run_name in RUN_NAMES:
    json_path = os.path.join(EXPERIMENTS_DIR, run_name, "fold_metrics.json")
    
    try:
        with open(json_path, 'r') as f:
            fold_data = json.load(f)
            
        # Extract the target metric across all 5 folds
        scores = [fold[TARGET_METRIC] for fold in fold_data]
        models[run_name] = scores
        print(f"Loaded {run_name}: {scores}")
        
    except FileNotFoundError:
        print(f"Error: Could not find {json_path}. Please check the run directory name.")
        exit()
    except KeyError:
        print(f"Error: The metric '{TARGET_METRIC}' was not found in the JSON file for {run_name}.")
        exit()

# --- 4. The Post-Hoc Test: Paired t-tests vs Rank 1 ---
print("\n--- Pairwise Paired t-tests vs Baseline (Holm Corrected) ---")

p_values_uncorrected = []
comparisons = []

# Isolate the baseline (first item) and the competitors
baseline_name = list(models.keys())[0]
baseline_scores = models[baseline_name]

print(f"Baseline Model: {baseline_name}\n")

for name, scores in list(models.items())[1:]:
    # Perform Paired t-test instead
    stat, p_val = stats.ttest_rel(baseline_scores, scores, alternative='two-sided')
    p_values_uncorrected.append(p_val)
    comparisons.append(name)
    
# Apply the zero-dependency Holm-Bonferroni correction
reject, p_values_corrected = holm_correction(p_values_uncorrected, alpha=0.05)

# Print Results
for comp, p_uncorr, p_corr, rej in zip(comparisons, p_values_uncorrected, p_values_corrected, reject):
    significance = "Significant difference" if rej else "No significant difference"
    print(f"vs {comp}:")
    print(f"   Uncorrected p: {p_uncorr:.5f} | Corrected p: {p_corr:.5f} -> {significance}")