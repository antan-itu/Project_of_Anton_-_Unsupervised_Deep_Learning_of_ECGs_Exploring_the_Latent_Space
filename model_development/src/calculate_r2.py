import os
import json
import numpy as np
import scipy.stats as st

# ==========================================
# 1. Run folders
# ==========================================
TARGET_FOLDERS = [
    "model_development/experiments/GridRun_001_2804_1735",
    "model_development/experiments/GridRun_001_2904_1242",
    "model_development/experiments/GridRun_003_2204_1304",
    "model_development/experiments/GridRun_002_2204_0913",
    "model_development/experiments/GridRun_001_2804_1029"
]

# ==========================================
# 2. Calculation Logic
# ==========================================
def calculate_ci(metric_list):
    n_folds = len(metric_list)
    mean_val = np.mean(metric_list)
    
    if n_folds <= 1:
        return f"[{mean_val:.3f}, {mean_val:.3f}]"
        
    std_err = np.std(metric_list, ddof=1) / np.sqrt(n_folds)
    margin_of_error = st.t.ppf(1 - 0.025, n_folds - 1) * std_err
    
    lower_bound = round(mean_val - margin_of_error, 3)
    upper_bound = round(mean_val + margin_of_error, 3)
    
    return f"[{lower_bound:.3f}, {upper_bound:.3f}]"

def main():
    for folder_path in TARGET_FOLDERS:
        run_name = os.path.basename(os.path.normpath(folder_path))
        metrics_file = os.path.join(folder_path, "fold_metrics.json")
        
        if not os.path.exists(metrics_file):
            print(f"{run_name}: Error - fold_metrics.json not found in this directory.")
            continue
            
        with open(metrics_file, 'r') as f:
            fold_metrics = json.load(f)
            
        val_r2_scores = [fold["Val_R2"] for fold in fold_metrics if "Val_R2" in fold]
        
        if not val_r2_scores:
            print(f"{run_name}: Error - No Val_R2 scores found in the metrics file.")
            continue
            
        ci_r2 = calculate_ci(val_r2_scores)
        
        print(f"{run_name} R2 95% CI: {ci_r2}")

if __name__ == "__main__":
    main()