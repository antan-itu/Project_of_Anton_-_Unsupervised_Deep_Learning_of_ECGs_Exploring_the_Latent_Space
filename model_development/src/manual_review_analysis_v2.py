import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

# 1. Define paths
csv_file_path = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/model_development/results/Manual_afib_review_results_with_predictions.CSV"
log_dir = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/model_development/results/logs"

# Ensure the log directory exists
os.makedirs(log_dir, exist_ok=True)

# 2. Load the data
df = pd.read_csv(csv_file_path, encoding='latin1', sep=';')
df.columns = df.columns.str.strip()

# 3. Clean and standardize the labels
df['Manual_AFib_Label'] = df['Manual_AFib_Label'].astype(str).str.lower()

# Create binary labels
df['Manual_Positive'] = df['Manual_AFib_Label'].str.contains('afib', na=False)
df['GT_Positive'] = df['GT_AFib_Label'].astype(str).str.strip() == '1'

# Extract Model Predictions
# Ensure the columns exist before using them
has_models = 'Model_XGB_Pred' in df.columns

if not has_models:
    print("Warning: Model prediction columns not found. Did you run the augmented CSV generation?")
else:
    # Ensure they are integers/floats for metric calculations
    df['Model_XGB_Pred'] = df['Model_XGB_Pred'].astype(int)
    df['Model_LR_Pred'] = df['Model_LR_Pred'].astype(int)
    df['Model_XGB_Prob'] = df['Model_XGB_Prob'].astype(float)
    df['Model_LR_Prob'] = df['Model_LR_Prob'].astype(float)


# 4. Calculate metrics (GT vs Manual)
total_samples = len(df)
total_manual_afib = df['Manual_Positive'].sum()
total_gt_afib = df['GT_Positive'].sum()

# Agreement metrics (GT vs Manual)
true_positives = ((df['GT_Positive'] == True) & (df['Manual_Positive'] == True)).sum()
true_negatives = ((df['GT_Positive'] == False) & (df['Manual_Positive'] == False)).sum()
total_agreement = true_positives + true_negatives

# Disagreement metrics (GT vs Manual)
false_positives = ((df['GT_Positive'] == True) & (df['Manual_Positive'] == False)).sum()
false_negatives = ((df['GT_Positive'] == False) & (df['Manual_Positive'] == True)).sum()

# Print text results
print("=" * 50)
print(f" BASELINE ANALYSIS: GT vs Manual Review ")
print("=" * 50)
print(f"Total ECGs reviewed: {total_samples}")
print(f"Total AFib in Manual Review: {total_manual_afib}")
print(f"Total AFib in MIMIC-IV GT: {total_gt_afib}")
print("-" * 30)

print(f"Total number of correctly labeled ECGs?")
print(f"Count: {total_agreement}")
print(f"Percentage: {(total_agreement / total_samples) * 100:.2f}%\n")

print(f"  -> Agree on AFib (True Positives): {true_positives}")
print(f"  -> Agree on No AFib (True Negatives): {true_negatives}\n")

print("-" * 30)
print(f"How many were false positives (GT=1, Manual=0)?")
print(f"Count: {false_positives}")
print(f"Percentage: {(false_positives / total_samples) * 100:.2f}%\n")

print(f"How many were false negatives (GT=0, Manual=1)?")
print(f"Count: {false_negatives}")
print(f"Percentage: {(false_negatives / total_samples) * 100:.2f}%\n")


# 5. Model Evaluation (Model vs Manual Review)
if has_models:
    print("\n" + "=" * 50)
    print(" MODEL PERFORMANCE vs MANUAL REVIEW (Gold Standard) ")
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
        
        # Calculate Advanced Metrics (Handling division by zero)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        try:
            auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
        except ValueError:
            auc = "N/A (Requires both classes)"
            pr_auc = "N/A"
            
        print(f"\n--- {model_name} ---")
        print(f"Accuracy:  {accuracy:.4f} ({(tp+tn)}/{total_samples} correct)")
        print(f"True Positives:  {tp}")
        print(f"True Negatives:  {tn}")
        print(f"False Positives: {fp} (Model said AFib, Manual said No)")
        print(f"False Negatives: {fn} (Model said No, Manual said AFib)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        if isinstance(auc, float):
             print(f"ROC-AUC:   {auc:.4f}")
             print(f"PR-AUC:    {pr_auc:.4f}")


# 6. Generate and save the Confusion Matrices
def plot_cm(y_true, y_pred, title, y_label, x_label, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No AFib', 'AFib'], 
                yticklabels=['No AFib', 'AFib'],
                annot_kws={"size": 14})
    plt.title(title, fontsize=14)
    plt.ylabel(y_label, fontsize=12)
    plt.xlabel(x_label, fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(log_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

print("\n--- Generating Plots ---")
# Baseline Plot
plot_cm(df['Manual_Positive'], df['GT_Positive'], 
        'Confusion Matrix: MIMIC-IV Labels vs Manual Review',
        'Manual Review (Gold Standard)', 'MIMIC-IV GT Label', 
        "afib_label_confusion_matrix.png")

# Model Plots
if has_models:
    plot_cm(df['Manual_Positive'], df['Model_XGB_Pred'], 
            'Confusion Matrix: XGBoost vs Manual Review',
            'Manual Review (Gold Standard)', 'XGBoost Prediction', 
            "afib_xgb_confusion_matrix.png")
            
    plot_cm(df['Manual_Positive'], df['Model_LR_Pred'], 
            'Confusion Matrix: Logistic Regression vs Manual Review',
            'Manual Review (Gold Standard)', 'LogReg Prediction', 
            "afib_lr_confusion_matrix.png")