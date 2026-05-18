import os
import json
import scipy.stats as stats

# --- 1. Configurations ---
EXPERIMENTS_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/model_development/experiments"

run_xgb = "GridRun_001_2804_1029"
run_lr = "GridRun_001_2804_1735"

# --- 2. Data Extraction ---
try:
    # Load XGB_PRAUC from the first run
    with open(os.path.join(EXPERIMENTS_DIR, run_xgb, "fold_metrics.json"), 'r') as f:
        xgb_scores = [fold["XGB_PRAUC"] for fold in json.load(f)]
    print(f"Loaded {run_xgb} (XGB_PRAUC): {xgb_scores}")

    # Load LR_PRAUC from the second run
    with open(os.path.join(EXPERIMENTS_DIR, run_lr, "fold_metrics.json"), 'r') as f:
        lr_scores = [fold["LR_PRAUC"] for fold in json.load(f)]
    print(f"Loaded {run_lr} (LR_PRAUC):  {lr_scores}")

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()
except KeyError as e:
    print(f"Error: Metric not found in JSON - {e}")
    exit()

# --- 3. Statistical Test ---
print("\n--- Paired t-test ---")

# Perform the test
stat, p_val = stats.ttest_rel(xgb_scores, lr_scores, alternative='two-sided')

print(f"t-statistic: {stat:.5f}")
print(f"p-value:     {p_val:.5f}")

if p_val < 0.05:
    winner = run_xgb if sum(xgb_scores) > sum(lr_scores) else run_lr
    print(f"\nResult: Significant difference found. {winner} is statistically better.")
else:
    print("\nResult: No significant difference between the two metrics.")