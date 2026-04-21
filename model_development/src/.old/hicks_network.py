import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# --- 1. SET DIRECTORIES ---
RUN_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/model_development/full_grid_search/GridRun_001_1003_1323"
FILE_PATH = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/data/full_training_set/training_dataset.h5"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- 2. LOAD LATENT VECTORS AND LABELS ---
print("Loading exported latent coordinates...")
X_latents = np.load(os.path.join(RUN_DIR, "saved_latents.npy"))
val_idx = np.load(os.path.join(RUN_DIR, "saved_val_idx.npy"))

print("Extracting exact AFib labels from Ground Truth...")
df_gt = pd.read_hdf(FILE_PATH, key='GT')

print("Aligning labels to HDF5 array index...")
df_gt = df_gt.sort_values(by='h5idx').reset_index(drop=True)

df_val_gt = df_gt.iloc[val_idx].copy()
report_cols = [f'report_{i}' for i in range(18)]

EXACT_TARGETS = [
    "ATRIAL FIBRILLATION",
    "Atrial fibrillation",
    "Atrial fibrillation."
]

mask = pd.Series(False, index=df_val_gt.index)
for col in report_cols:
    if col in df_val_gt.columns:
        mask |= df_val_gt[col].fillna('').astype(str).str.strip().isin(EXACT_TARGETS)

# Ground Truth = 1 if AFib is present, 0 if not
y_labels = mask.astype(int).values

print(f"Total Samples: {len(y_labels)}")
print(f"AFib Positive (1): {sum(y_labels)}")
print(f"AFib Negative (0): {len(y_labels) - sum(y_labels)}")

# --- 3. TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X_latents, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

# Convert to PyTorch Tensors
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --- 4. BUILD THE FEEDFORWARD NETWORK ---
class LatentClassifier(nn.Module):
    def __init__(self, input_dim):
        super(LatentClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

model = LatentClassifier(input_dim=512).to(DEVICE)

pos_weight_val = (len(y_labels) - sum(y_labels)) / sum(y_labels)
weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# --- 5. TRAINING LOOP ---
print("\nTraining Downstream Classifier on Latent Space...")
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {running_loss/len(train_loader):.4f}")

# --- 6. EVALUATION ---
print("\nEvaluating Classifier...")
model.eval()
y_true = []
y_pred_probs = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(DEVICE)
        logits = model(x_batch)
        probs = torch.sigmoid(logits) 
        y_true.extend(y_batch.numpy().flatten())
        y_pred_probs.extend(probs.cpu().numpy().flatten())

y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs)
y_pred_labels = (y_pred_probs >= 0.5).astype(int)

# --- 7. PRINT METRICS ---
print("\n" + "="*40)
print("DOWNSTREAM CLASSIFICATION RESULTS")
print("="*40)
print(f"Accuracy:  {accuracy_score(y_true, y_pred_labels):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_true, y_pred_probs):.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred_labels, target_names=["Not AFib", "AFib"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_labels))
print("="*40)