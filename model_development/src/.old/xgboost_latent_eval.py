import os
import numpy as np
import pandas as pd
import h5py
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# --- 1. SET DIRECTORIES ---
BASE_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space"
RUN_DIR = os.path.join(BASE_DIR, "model_development/experiments/GridRun_001_1104_1018")
DATA_PATH = os.path.join(BASE_DIR, "data/MIMIC_IV_ECG_HDF5/mimic_iv_train.h5")
PLOT_DIR = os.path.join(RUN_DIR, "plots")

os.makedirs(PLOT_DIR, exist_ok=True)

# --- 2. LOAD LATENTS ---
print("Loading FULL exported latent coordinates...")
X_latents = np.load(os.path.join(RUN_DIR, "FULL_latents.npy"))

# --- 3. REBUILD LABELS ON THE FLY ---
print("Extracting exact AFib labels from Ground Truth HDF5...")
df_gt_dict = {}
with h5py.File(DATA_PATH, 'r') as f:
    gt_group = f['GT']
    report_cols = [key for key in gt_group.keys() if key.startswith('report_')]
    for col in report_cols:
        df_gt_dict[col] = [val.decode('utf-8') for val in gt_group[col][:]]

df_gt = pd.DataFrame(df_gt_dict)
EXACT_TARGETS = ["ATRIAL FIBRILLATION", "Atrial fibrillation", "Atrial fibrillation."]

mask = pd.Series(False, index=df_gt.index)
for col in report_cols:
    if col in df_gt.columns:
        mask |= df_gt[col].fillna('').astype(str).str.strip().isin(EXACT_TARGETS)

y_labels = mask.astype(int).values

num_positive = sum(y_labels)
num_negative = len(y_labels) - num_positive

print(f"Total Samples: {len(y_labels)}")
print(f"AFib Positive (1): {num_positive}")
print(f"AFib Negative (0): {num_negative}")

# Safety check to ensure matrices align perfectly
assert len(X_latents) == len(y_labels), "Mismatch between latents and labels!"

# --- 4. TRAIN/TEST SPLIT ---
print("\nSplitting data into 80/20 Train/Test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_latents, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

# Calculate the optimal weight for the minority class
optimal_weight = num_negative / num_positive
print(f"\nCalculated scale_pos_weight for XGBoost: {optimal_weight:.2f}")

# --- 5. BUILD AND TRAIN XGBOOST ---
print("\nTraining XGBoost Classifier on Latent Space...")
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=optimal_weight, # Forces the model to care about the minority class
    tree_method='hist',              # Ultra-fast training for massive datasets
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1                        # Uses all available CPU cores
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# --- 6. EVALUATION ---
print("\nEvaluating Classifier...")
y_pred_probs = xgb_model.predict_proba(X_test)[:, 1]
y_pred_labels = xgb_model.predict(X_test)

# --- 7. PRINT METRICS ---
print("\n" + "="*40)
print("XGBOOST CLASSIFICATION RESULTS")
print("="*40)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_labels):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_probs):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_labels, target_names=["Not AFib", "AFib"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_labels))
print("="*40)

# --- 8. FEATURE IMPORTANCE ---
print("\nExtracting Top 15 Latent Dimensions...")
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1][:15] # Get top 15

plt.figure(figsize=(10, 6))
plt.title("Top 15 Most Important Latent Dimensions for AFib")
plt.bar(range(15), importances[indices], align="center", color='#1d3557')
plt.xticks(range(15), [f"Dim {i}" for i in indices], rotation=45)
plt.xlim([-1, 15])
plt.ylabel("Relative Importance")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

save_path = os.path.join(PLOT_DIR, "xgboost_feature_importance.png")
plt.savefig(save_path, dpi=300)
print(f"Saved Feature Importance plot to: {save_path}")