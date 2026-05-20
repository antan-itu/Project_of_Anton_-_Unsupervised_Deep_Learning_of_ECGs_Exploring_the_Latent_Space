import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 1. Define paths
csv_file_path = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/model_development/results/Manual_afib_review_results.CSV"
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

# 4. Calculate metrics
total_samples = len(df)
total_manual_afib = df['Manual_Positive'].sum()
total_gt_afib = df['GT_Positive'].sum()

# Agreement metrics
true_positives = ((df['GT_Positive'] == True) & (df['Manual_Positive'] == True)).sum()
true_negatives = ((df['GT_Positive'] == False) & (df['Manual_Positive'] == False)).sum()
total_agreement = true_positives + true_negatives  # This is the same as correctly_labeled

# Disagreement metrics
false_positives = ((df['GT_Positive'] == True) & (df['Manual_Positive'] == False)).sum()
false_negatives = ((df['GT_Positive'] == False) & (df['Manual_Positive'] == True)).sum()

# Print text results
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
print("-" * 30)

# 5. Generate and save the Confusion Matrix
# Treating Manual Review as True and GT as Predicted (MIMIC-IV label)
y_true = df['Manual_Positive']
y_pred = df['GT_Positive']

cm = confusion_matrix(y_true, y_pred)

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No AFib', 'AFib'], 
            yticklabels=['No AFib', 'AFib'],
            annot_kws={"size": 14})

plt.title('Confusion Matrix: MIMIC-IV Labels vs Manual Review', fontsize=14)
plt.ylabel('Manual Review (Jørgen)', fontsize=12)
plt.xlabel('MIMIC-IV GT Label', fontsize=12)

# Save the plot
save_path = os.path.join(log_dir, "afib_label_confusion_matrix.png")
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"Confusion matrix saved successfully to:\n{save_path}")