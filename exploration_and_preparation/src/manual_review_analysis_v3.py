### The script analyse whether MIMIC and the models align with the manual review
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, cohen_kappa_score
from sklearn.calibration import calibration_curve

# Define paths
csv_file_path = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/model_development/results/Manual_af_review_results_with_predictions.CSV"
log_dir = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/model_development/results/logs"
os.makedirs(log_dir, exist_ok=True)

# Load the data
df = pd.read_csv(csv_file_path, encoding='latin1', sep=';')
df.columns = df.columns.str.strip()

# Clean and standardize the labels
df['Manual_AF_Label'] = df['Manual_AF_Label'].astype(str).str.lower()

# Create binary labels
df['Manual_Positive'] = df['Manual_AF_Label'].str.contains('af', na=False)
df['GT_Positive'] = df['GT_AF_Label'].astype(str).str.strip() == '1'

# Extract Model Predictions
has_models = 'Model_XGB_Pred' in df.columns

if not has_models:
    print("Warning: Model prediction columns not found. Did you run the augmented CSV generation?")
else:
    # Ensure they are integers/floats for metric calculations
    df['Model_XGB_Pred'] = df['Model_XGB_Pred'].astype(int)
    df['Model_LR_Pred'] = df['Model_LR_Pred'].astype(int)
    df['Model_XGB_Prob'] = df['Model_XGB_Prob'].astype(float)
    df['Model_LR_Prob'] = df['Model_LR_Prob'].astype(float)

# Calculate metrics (GT vs Manual)
total_samples = len(df)
total_manual_af = df['Manual_Positive'].sum()
total_gt_af = df['GT_Positive'].sum()

# Agreement metrics (GT vs Manual)
true_positives = ((df['GT_Positive'] == True) & (df['Manual_Positive'] == True)).sum()
true_negatives = ((df['GT_Positive'] == False) & (df['Manual_Positive'] == False)).sum()
total_agreement = true_positives + true_negatives

# Disagreement metrics (GT vs Manual)
false_positives = ((df['GT_Positive'] == True) & (df['Manual_Positive'] == False)).sum()
false_negatives = ((df['GT_Positive'] == False) & (df['Manual_Positive'] == True)).sum()

# Kappa Score
baseline_kappa = cohen_kappa_score(df['Manual_Positive'], df['GT_Positive'])

# Print results
print("=" * 50)
print(f" Baseline: GT vs Manual Review ")
print("=" * 50)
print(f"Total ECGs reviewed: {total_samples}")
print(f"Total AF in Manual Review: {total_manual_af}")
print(f"Total AF in MIMIC GT: {total_gt_af}")
print("-" * 30)

print(f"Total number of correctly labeled ECGs?")
print(f"Count: {total_agreement}")
print(f"Percentage: {(total_agreement / total_samples) * 100:.2f}%\n")

print(f"  -> Agree on AF in Manual Review (True Positives): {true_positives}")
print(f"  -> Agree on No AF in Manual Review (True Negatives): {true_negatives}\n")

print("-" * 30)
print(f"How many were false positives (GT=1, Manual=0)?")
print(f"Count: {false_positives}")
print(f"Percentage: {(false_positives / total_samples) * 100:.2f}%\n")

print(f"How many were false negatives (GT=0, Manual=1)?")
print(f"Count: {false_negatives}")
print(f"Percentage: {(false_negatives / total_samples) * 100:.2f}%\n")

print(f"Cohen's Kappa Score: {baseline_kappa:.4f}")

# Models vs Manual Review
if has_models:
    print("\n" + "=" * 50)
    print(" Modes vs Manual review ")
    print("=" * 50)

    # Manual_Positive as the absolute truth
    y_true = df['Manual_Positive'].astype(int)
    
    models = {
        'XGBoost': {'pred': df['Model_XGB_Pred'], 'prob': df['Model_XGB_Prob']},
        'Logistic Regression': {'pred': df['Model_LR_Pred'], 'prob': df['Model_LR_Prob']}
    }
    
    for model_name, data in models.items():
        y_pred = data['pred']
        y_prob = data['prob']
        
        # Calculate Confusion Matrix elements
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        accuracy = (tp + tn) / total_samples
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        kappa = cohen_kappa_score(y_true, y_pred)
        
        try:
            auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
        except ValueError:
            auc = "N/A (Requires both classes)"
            pr_auc = "N/A"
            
        print(f"\n--- {model_name} ---")
        print(f"Accuracy:  {accuracy:.4f} ({(tp+tn)}/{total_samples} correct)")
        print(f"Cohen's Kappa: {kappa:.4f}")
        print(f"True Positives:  {tp}")
        print(f"True Negatives:  {tn}")
        print(f"False Positives: {fp} (Model said AF, Manual said No)")
        print(f"False Negatives: {fn} (Model said No, Manual said AF)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        if isinstance(auc, float):
             print(f"ROC-AUC:   {auc:.4f}")
             print(f"PR-AUC:    {pr_auc:.4f}")

# Save plots
def plot_cm(y_true, y_pred, title, y_label, x_label, filename):
    cm = confusion_matrix(y_true, y_pred)
    
    # Create text labels for the matrix
    group_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                xticklabels=['No AF', 'AF'], 
                yticklabels=['No AF', 'AF'],
                annot_kws={"size": 16})
                
    plt.title(title, fontsize=16)
    plt.ylabel(y_label, fontsize=14)
    plt.xlabel(x_label, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(log_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

def plot_calibration_bars(y_true, y_prob, title, filename, n_bins=10):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    plt.figure(figsize=(8, 6))
    
    # Plot ideal calibration line
    plt.plot([0, 1], [0, 1], color='#4B8B3B', linewidth=2.5, label='Perfectly Calibrated', zorder=1)
    
    # Plot bars
    bar_width = 1.0 / n_bins
    plt.bar(mean_predicted_value, fraction_of_positives, width=bar_width*0.8, 
            alpha=0.6, color='#A1C9F4', edgecolor='#1F77B4', linewidth=1.5, 
            label='Model Accuracy', zorder=2)
    
    plt.xlabel('Model Confidence (Predicted Probability)', fontsize=15)
    plt.ylabel('Accuracy (Fraction of Positives)', fontsize=15)
    plt.title(title, fontsize=17)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=12)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--', zorder=0)
    plt.tight_layout()
    
    save_path = os.path.join(log_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

print("\n Generating plots ")
# Baseline Plot
plot_cm(df['Manual_Positive'], df['GT_Positive'], 
        'Confusion Matrix: MIMIC-IV Labels vs Manual Review',
        'Manual Review', 'MIMIC-IV GT Label', 
        "af_label_confusion_matrix.png")

# Model plots
if has_models:
    # Confusion Matrices
    plot_cm(df['Manual_Positive'], df['Model_XGB_Pred'], 
            'Confusion Matrix: XGBoost vs Manual Review',
            'Manual Review ', 'XGBoost Prediction', 
            "af_xgb_confusion_matrix.png")
            
    plot_cm(df['Manual_Positive'], df['Model_LR_Pred'], 
            'Confusion Matrix: Logistic Regression vs Manual Review',
            'Manual Review', 'LogReg Prediction', 
            "af_lr_confusion_matrix.png")
            
    # Calibration Plots
    plot_calibration_bars(df['Manual_Positive'].astype(int), df['Model_XGB_Prob'],
                          'Calibration Curve: XGBoost',
                          'af_xgb_calibration_plot.png')
                          
    plot_calibration_bars(df['Manual_Positive'].astype(int), df['Model_LR_Prob'],
                          'Calibration Curve: Logistic Regression',
                          'af_lr_calibration_plot.png')
    

# Model agreement (XGBoost vs Logistic regression)
if has_models:
    print("\n" + "=" * 50)
    print(" MODEL AGREEMENT ANALYSIS ")
    print("=" * 50)

    actual_af_mask = df['Manual_Positive'] == True
    actual_no_af_mask = df['Manual_Positive'] == False
    
    xgb_pred_pos = df['Model_XGB_Pred'] == 1
    lr_pred_pos = df['Model_LR_Pred'] == 1

    # True Positives Overlap (Actual AF cases) ---
    both_found = (actual_af_mask & xgb_pred_pos & lr_pred_pos).sum()
    only_xgb_found = (actual_af_mask & xgb_pred_pos & ~lr_pred_pos).sum()
    only_lr_found = (actual_af_mask & ~xgb_pred_pos & lr_pred_pos).sum()
    neither_found = (actual_af_mask & ~xgb_pred_pos & ~lr_pred_pos).sum()

    print("--- True AF cases (Overlap in true positives) ---")
    print(f"Found by BOTH models: {both_found}")
    print(f"Found ONLY by XGBoost: {only_xgb_found}")
    print(f"Found ONLY by LogReg: {only_lr_found}")
    print(f"Missed by BOTH models: {neither_found}")

    # False positives overlap ---
    both_fp = (actual_no_af_mask & xgb_pred_pos & lr_pred_pos).sum()
    only_xgb_fp = (actual_no_af_mask & xgb_pred_pos & ~lr_pred_pos).sum()
    only_lr_fp = (actual_no_af_mask & ~xgb_pred_pos & lr_pred_pos).sum()

    print("\n--- False positives (Overlap) ---")
    print(f"Both models incorrectly predicted AF: {both_fp}")
    print(f"Only XGBoost incorrectly predicted AF: {only_xgb_fp}")
    print(f"Only LogReg incorrectly predicted AF: {only_lr_fp}")
    
    # --- 3. Print the difficult cases for review ---
    missed_by_both_indices = df[actual_af_mask & ~xgb_pred_pos & ~lr_pred_pos]['HDF5_Index']
    if not missed_by_both_indices.empty:
        print(f"\nIndices of AF cases missed by BOTH models:\n{missed_by_both_indices.tolist()}")

    both_fp_indices = df[actual_no_af_mask & xgb_pred_pos & lr_pred_pos]['HDF5_Index']
    if not both_fp_indices.empty:
        print(f"\nIndices of false positives predicted by BOTH models:\n{both_fp_indices.tolist()}")