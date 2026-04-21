# ================================
# 1. IMPORTS & GLOBAL CONFIG
# ================================
import os
import gc
import math
import h5py
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import umap.umap_ as umap
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ================================
# 2. DIRECTORIES & PATHS
# ================================
BASE_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space"
RUN_DIR = os.path.join(BASE_DIR, "model_development/experiments/GridRun_002_1704_0250") 
DATA_PATH = os.path.join(BASE_DIR, "data/MIMIC_IV_ECG_HDF5/mimic_iv_train.h5")
PLOT_DIR = os.path.join(RUN_DIR, "plots_posthoc") # Saving in a distinct folder to avoid overwrites

os.makedirs(PLOT_DIR, exist_ok=True)

# ================================
# 3. PYTORCH CLASSES (Required for loading model)
# ================================
class ECGDataset:
    def __init__(self, h5_file_path):
        print(f"\nLoading entire dataset directly into SYSTEM RAM from {h5_file_path}...")
        with h5py.File(h5_file_path, 'r') as h5f:
            self.data = torch.tensor(h5f['rhythm_filtered'][:], dtype=torch.float32).permute(0, 2, 1)
        print(f"Dataset loaded to CPU RAM. Shape: {self.data.shape}")
        
        print("Standardizing data (In-Place)...")
        means = self.data.mean(dim=2, keepdim=True)
        stds = self.data.std(dim=2, keepdim=True)
        self.data -= means
        self.data /= (stds + 1e-8)
        
        del means, stds
        gc.collect()
        print("Data standardized and ready.")

class FastTensorDataLoader:
    def __init__(self, dataset, indices, batch_size, shuffle=False):
        self.dataset = dataset
        self.indices = torch.tensor(indices, dtype=torch.long)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = math.ceil(len(self.indices) / self.batch_size)
        
    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(len(self.indices))
            self.indices = self.indices[perm]
        self.current_batch = 0
        return self
        
    def __next__(self):
        if self.current_batch >= self.n_batches:
            raise StopIteration
        start = self.current_batch * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        batch_idx = self.indices[start:end]
        
        x_batch = self.dataset.data[batch_idx].to(DEVICE)
        self.current_batch += 1
        return x_batch, x_batch 
        
    def __len__(self):
        return self.n_batches

class ConvAutoencoder(nn.Module):
    # (Pasting the exact class from Script 1)
    def __init__(self, seq_len, in_channels, latent_dim, base_filters, kernel_size,
                 num_layers, pool_size, activation, dropout_rate, norm_type, pooling_type, masking_ratio=0.0):
        super(ConvAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.masking_ratio = masking_ratio 
        padding = kernel_size // 2
        
        encoder_layers = []
        current_channels = in_channels
        for i in range(num_layers):
            filters = base_filters * (2**i)
            conv_stride = pool_size if pooling_type == 'stride' else 1
            encoder_layers.append(nn.Conv1d(current_channels, filters, kernel_size, stride=conv_stride, padding=padding))
            if norm_type == 'layer':
                encoder_layers.append(nn.GroupNorm(1, filters))
            elif norm_type == 'batch':
                encoder_layers.append(nn.BatchNorm1d(filters))
            if activation == 'leaky_relu':
                encoder_layers.append(nn.LeakyReLU())
            else:
                encoder_layers.append(nn.ReLU())
            if pooling_type == 'max':
                encoder_layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
            elif pooling_type == 'average':
                encoder_layers.append(nn.AvgPool1d(kernel_size=pool_size, stride=pool_size))
            if dropout_rate > 0.0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            current_channels = filters
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        dummy_input = torch.zeros(1, in_channels, seq_len)
        dummy_output = self.encoder(dummy_input)
        self.shape_before_flatten = dummy_output.shape[1:]
        flattened_size = int(np.prod(self.shape_before_flatten))
        
        self.fc_latent = nn.Linear(flattened_size, latent_dim)
        self.fc_decoder_input = nn.Linear(latent_dim, flattened_size)
        
        decoder_layers = []
        if activation == 'leaky_relu':
            decoder_layers.append(nn.LeakyReLU())
        else:
            decoder_layers.append(nn.ReLU())
            
        for i in reversed(range(num_layers)):
            filters = base_filters * (2**i)
            out_channels_next = base_filters * (2**(i-1)) if i > 0 else in_channels
            if pooling_type in ['max', 'average']:
                decoder_layers.append(nn.Upsample(scale_factor=pool_size))
                conv_stride = 1
            else:
                conv_stride = pool_size
            decoder_layers.append(nn.ConvTranspose1d(current_channels, out_channels_next, kernel_size, 
                                                     stride=conv_stride, padding=padding, output_padding=conv_stride-1 if conv_stride > 1 else 0))
            if i > 0: 
                if norm_type == 'layer':
                    decoder_layers.append(nn.GroupNorm(1, out_channels_next))
                elif norm_type == 'batch':
                    decoder_layers.append(nn.BatchNorm1d(out_channels_next))
                if activation == 'leaky_relu':
                    decoder_layers.append(nn.LeakyReLU())
                else:
                    decoder_layers.append(nn.ReLU())
                if dropout_rate > 0.0:
                    decoder_layers.append(nn.Dropout(dropout_rate))
            current_channels = out_channels_next

        self.decoder = nn.Sequential(*decoder_layers)
        self.final_conv = nn.Conv1d(in_channels, in_channels, 1)

    def forward(self, x):
        if self.training and self.masking_ratio > 0.0:
            mask = (torch.rand_like(x) > self.masking_ratio).float()
            x_input = x * mask
        else:
            x_input = x

        encoded = self.encoder(x_input)
        flattened = encoded.view(encoded.size(0), -1)
        latent = self.fc_latent(flattened)
        decoded_input = self.fc_decoder_input(latent)
        reshaped = decoded_input.view(decoded_input.size(0), *self.shape_before_flatten)
        decoded = self.decoder(reshaped)
        
        if decoded.size(2) > self.seq_len:
            decoded = decoded[:, :, :self.seq_len]
        elif decoded.size(2) < self.seq_len:
            pad_size = self.seq_len - decoded.size(2)
            decoded = torch.nn.functional.pad(decoded, (0, pad_size))
            
        out = self.final_conv(decoded)
        return out, latent


# ================================
# 4. DATA LOADING & LATENT EXTRACTION
# ================================
# Load the raw dataset
full_dataset = ECGDataset(h5_file_path=DATA_PATH)

# Load indices
print("Loading train and val indices...")
train_idx = np.load(os.path.join(RUN_DIR, "saved_train_idx.npy"))
val_idx = np.load(os.path.join(RUN_DIR, "saved_val_idx.npy"))

# Load Model
print("Loading PyTorch Autoencoder Model...")
model = ConvAutoencoder(
    seq_len=5000, in_channels=8, latent_dim=64, base_filters=64, kernel_size=9,
    num_layers=3, pool_size=3, activation='leaky_relu', dropout_rate=0, 
    norm_type='batch', pooling_type='average', masking_ratio=0
).to(DEVICE)

model.load_state_dict(torch.load(os.path.join(RUN_DIR, "best_fold_model.pth"), weights_only=True))
model.eval()

# Latent Extraction Function
def extract_latents(dataset, indices, batch_size=512):
    loader = FastTensorDataLoader(dataset, indices, batch_size, shuffle=False)
    latents_list = []
    with torch.no_grad():
        for xb, _ in loader:
            _, latents = model(xb)
            latents_list.append(latents.cpu().numpy())
    return np.concatenate(latents_list, axis=0)

print("\nExtracting Training Latents...")
X_train = extract_latents(full_dataset, train_idx)
print("Extracting Validation (Test) Latents...")
X_test = extract_latents(full_dataset, val_idx)

# ================================
# 5. BUILD GLOBAL LABELS & SNIPPETS
# ================================
print("\nExtracting exact AFib labels and clinical text from Ground Truth HDF5...")
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

y_labels_global = mask.astype(int).values

# Build hover snippets for later plotting
combined_reports = df_gt[report_cols].fillna('').astype(str).agg(' '.join, axis=1)
clean_reports = combined_reports.str.strip().str.replace(r'\s+', ' ', regex=True)
global_hover_snippets = clean_reports.str.slice(0, 250) + "..."

# Slice the labels based on exact indices
y_train = y_labels_global[train_idx]
y_test = y_labels_global[val_idx]
test_hover_snippets = global_hover_snippets.iloc[val_idx].values

# Calculate weights based ONLY on the training distribution
num_positive = sum(y_train)
num_negative = len(y_train) - num_positive
optimal_weight = num_negative / num_positive if num_positive > 0 else 1.0

print(f"\nCalculated scale_pos_weight for XGBoost (from Training set): {optimal_weight:.2f}")
print(f"Validation Samples (Holdout Test Set): {len(y_test)} (AFib: {sum(y_test)})")

# ================================
# 6. XGBOOST TRAINING & EVALUATION
# ================================
print("\nTraining XGBoost Classifier on Latent Space...")
# Reverted hyperparameters to match Script 1 to guarantee identical AUC
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=optimal_weight, 
    tree_method='hist',              
    n_estimators=150,           # Matched to Script 1
    learning_rate=0.05,         # Matched to Script 1
    max_depth=5,                # Matched to Script 1
    eval_metric='auc',
    random_state=42,
    n_jobs=-1                        
)

xgb_model.fit(X_train, y_train)

print("\nEvaluating Classifier...")
y_pred_probs = xgb_model.predict_proba(X_test)[:, 1]
y_pred_labels = xgb_model.predict(X_test)

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

# Feature Importance
importances = xgb_model.feature_importances_
indices_imp = np.argsort(importances)[::-1][:15] 

plt.figure(figsize=(10, 6))
plt.title("Top 15 Most Important Latent Dimensions for AFib")
plt.bar(range(15), importances[indices_imp], align="center", color='#1d3557')
plt.xticks(range(15), [f"Dim {i}" for i in indices_imp], rotation=45)
plt.xlim([-1, 15])
plt.ylabel("Relative Importance")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "xgboost_feature_importance.png"), dpi=300)
plt.close()

# =====================================================================
# 7. ERROR ANALYSIS: FALSE NEGATIVE PLOTS
# =====================================================================
print("\nHunting down False Negatives (Missed AFib cases)...")

fn_mask = (y_test == 1) & (y_pred_labels == 0)
fn_local_indices = np.where(fn_mask)[0]

# Mapping directly to the HDF5 index since X_test == val_idx
original_fn_indices = val_idx[fn_local_indices]
print(f"Found {len(original_fn_indices)} False Negatives. Plotting the first 25...")

num_to_plot = min(25, len(original_fn_indices))
plot_indices = original_fn_indices[:num_to_plot]

fn_save_dir = os.path.join(PLOT_DIR, 'false_negatives_8_lead_batch')
os.makedirs(fn_save_dir, exist_ok=True)

lead_names = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

with h5py.File(DATA_PATH, 'r') as h5f:
    dataset = h5f['rhythm_filtered']
    for count, idx in enumerate(plot_indices):
        signal = dataset[idx]
        patient_diag = global_hover_snippets.iloc[idx]
        if signal.shape[0] > signal.shape[1]:
            signal = signal.T 
            
        fig, axes = plt.subplots(8, 1, figsize=(15, 12), sharex=True)
        fig.suptitle(f"Missed AFib (False Negative) | HDF5 Index: {idx}\nDiag: {patient_diag}", fontsize=14)
        
        for lead_idx in range(8):
            ax = axes[lead_idx]
            ax.plot(signal[lead_idx], color='#1d3557', linewidth=0.8)
            label = lead_names[lead_idx]
            ax.set_ylabel(label, fontsize=10, rotation=0, labelpad=20, ha='right')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_yticks([])
            
        axes[-1].set_xlabel('Time (Samples)', fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = os.path.join(fn_save_dir, f"fn_8_lead_idx_{idx}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

# =====================================================================
# 8. ERROR ANALYSIS: THEMED UMAP PROJECTION
# =====================================================================
print("\nRunning UMAP to map where the False Negatives hide...")

N_NEIGHBORS = 25
MIN_DIST = 0.01

print(f"Calculating 2D UMAP (Neighbors: {N_NEIGHBORS}, Min Dist: {MIN_DIST})...")
reducer_2d = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, n_components=2, random_state=42)
test_emb_2d = reducer_2d.fit_transform(X_test)

print("Calculating 3D UMAP Projection...")
reducer_3d = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, n_components=3, random_state=42)
test_emb_3d = reducer_3d.fit_transform(X_test)

category_labels = []
for true_lbl, pred_lbl in zip(y_test, y_pred_labels):
    if true_lbl == 1 and pred_lbl == 1: category_labels.append('Detected AFib (TP)')
    elif true_lbl == 1 and pred_lbl == 0: category_labels.append('Missed AFib (FN)')
    elif true_lbl == 0 and pred_lbl == 1: category_labels.append('False Alarm (FP)')
    else: category_labels.append('Normal / Other (TN)')

custom_palette = {
    'Missed AFib (FN)': 'red', 'Detected AFib (TP)': '#457b9d', 
    'False Alarm (FP)': 'orange', 'Normal / Other (TN)': 'lightgrey'
}

# 2D Plot
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=test_emb_2d[:, 0], y=test_emb_2d[:, 1], hue=category_labels,
    palette=custom_palette, alpha=0.6, s=15, edgecolor=None
)
plt.title(f'2D Error Analysis UMAP (n_neighbors={N_NEIGHBORS})', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_path_2d = os.path.join(PLOT_DIR, "umap_error_analysis_2d.png")
plt.savefig(save_path_2d, dpi=300)
plt.close()

# 3D Interactive Plot
plot_df = pd.DataFrame({
    'UMAP_3D_1': test_emb_3d[:, 0], 'UMAP_3D_2': test_emb_3d[:, 1], 'UMAP_3D_3': test_emb_3d[:, 2],
    'Diagnosis': category_labels, 'Report_Snippet': test_hover_snippets
})
plot_df['sort_order'] = plot_df['Diagnosis'].map({'Normal / Other (TN)': 0, 'False Alarm (FP)': 1, 'Detected AFib (TP)': 2, 'Missed AFib (FN)': 3})
plot_df = plot_df.sort_values(by='sort_order')

fig_3d = px.scatter_3d(
    plot_df, x='UMAP_3D_1', y='UMAP_3D_2', z='UMAP_3D_3',
    color='Diagnosis', color_discrete_map=custom_palette, opacity=0.6,
    hover_data={'Report_Snippet': True, 'UMAP_3D_1': False, 'UMAP_3D_2': False, 'UMAP_3D_3': False, 'sort_order': False},
    title=f'Interactive 3D Error Analysis (n_neighbors={N_NEIGHBORS}): Missed AFib Cases'
)
fig_3d.update_traces(marker=dict(size=3, line=dict(width=0)))
fig_3d.update_layout(
    scene=dict(xaxis=dict(showbackground=False), yaxis=dict(showbackground=False), zaxis=dict(showbackground=False)),
    margin=dict(r=10, b=10, l=10, t=40)
)
save_path_3d = os.path.join(PLOT_DIR, "umap_error_analysis_3d_interactive.html")
fig_3d.write_html(save_path_3d)

print(f"\nSaved Plots to: {PLOT_DIR}")
print("Script finished executing.")