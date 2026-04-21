import os
import h5py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# --- 1. SET DIRECTORIES ---
BASE_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space"
RUN_DIR = os.path.join(BASE_DIR, "model_development/experiments/GridRun_001_1104_1018")
DATA_PATH = os.path.join(BASE_DIR, "data/MIMIC_IV_ECG_HDF5/mimic_iv_train.h5")

# --- 2. LOAD LATENTS ---
print("Loading FULL exported latent coordinates...")
X_latents = np.load(os.path.join(RUN_DIR, "FULL_latents.npy"))

# --- 3. REBUILD LABELS ON THE FLY ---
print("Extracting exact AFib labels from Ground Truth...")
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

# Double check that the arrays match!
assert len(X_latents) == len(y_labels), "Mismatch between latents and labels!"

# --- 4. TRAIN/TEST SPLIT ---
print("\nSplitting data into 80/20 Train/Test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_latents, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

# --- 5. BUILD AND TRAIN LOGISTIC REGRESSION ---
print("\nTraining Linear Probe (Logistic Regression) on Latent Space...")
linear_model = LogisticRegression(
    class_weight='balanced', 
    max_iter=1000, 
    random_state=42,
    n_jobs=-1 
)

linear_model.fit(X_train, y_train)

# --- 6. EVALUATION ---
print("\nEvaluating Classifier...")
y_pred_probs = linear_model.predict_proba(X_test)[:, 1]
y_pred_labels = linear_model.predict(X_test)

# --- 7. PRINT METRICS ---
print("\n" + "="*50)
print("LINEAR PROBE (LOGISTIC REGRESSION) RESULTS")
print("="*50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_labels):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_probs):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_labels, target_names=["Not AFib", "AFib"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_labels))
print("="*50)