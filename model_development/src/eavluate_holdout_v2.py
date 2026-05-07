import os
import json
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import h5py
import gc
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

# ================================
# 1. Configuration & Paths
# ================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

BASE_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space"
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data/MIMIC_IV_ECG_HDF5/mimic_iv_train.h5")
HOLDOUT_DATA_PATH = os.path.join(BASE_DIR, "data/MIMIC_IV_ECG_HDF5/mimic_iv_holdout.h5")

RUN_DIR = os.path.join(BASE_DIR, "model_development/experiments/GridRun_001_2804_1735")
CONFIG_PATH = os.path.join(RUN_DIR, "config.json")
MODEL_WEIGHTS_PATH = os.path.join(RUN_DIR, "best_fold_model.pth")

EXACT_TARGETS = ["ATRIAL FIBRILLATION", "Atrial fibrillation", "Atrial fibrillation."]

# ================================
# 2. Classes
# ================================
class ECGDataset:
    def __init__(self, h5_file_path, seq_len):
        print(f"Loading dataset into RAM from {h5_file_path}...")
        with h5py.File(h5_file_path, 'r') as h5f:
            self.data = torch.tensor(h5f['rhythm_filtered'][:], dtype=torch.float32).permute(0, 2, 1)[:, :, :seq_len]
        
        print("Standardizing data (In-Place)...")
        means = self.data.mean(dim=2, keepdim=True)
        stds = self.data.std(dim=2, keepdim=True)
        self.data -= means
        self.data /= (stds + 1e-8)
        del means, stds
        gc.collect()

class FastTensorDataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.indices = torch.arange(len(dataset.data), dtype=torch.long)
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
        return x_batch

class ConvAutoencoder(nn.Module):
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
# 3. Helper Functions
# ================================
def extract_afib_labels(h5_file_path):
    """Extracts binary AFib labels directly from the HDF5 metadata."""
    df_gt_dict = {}
    with h5py.File(h5_file_path, 'r') as f:
        gt_group = f['GT']
        report_cols = [key for key in gt_group.keys() if key.startswith('report_')]
        for col in report_cols:
            df_gt_dict[col] = [val.decode('utf-8') for val in gt_group[col][:]]

    df_gt = pd.DataFrame(df_gt_dict)
    mask = pd.Series(False, index=df_gt.index)
    for col in report_cols:
        if col in df_gt.columns:
            mask |= df_gt[col].fillna('').astype(str).str.strip().isin(EXACT_TARGETS)
    return mask.astype(int).values

def get_latents_and_reconstruction(model, dataloader):
    """Extracts latents and per-sample sum of squares for rapid bootstrapping."""
    model.eval()
    latents_list, ss_res_list, ss_tot_list = [], [], []
    
    with torch.no_grad():
        for x_batch in dataloader:
            outputs, latents = model(x_batch)
            latents_list.append(latents.cpu().numpy())
            
            # Calculate sum of squares per-sample across channels and sequence length
            # Dim: (Batch, Channels, SeqLen) -> sum over dims 1 and 2
            y_true, y_pred = x_batch, outputs
            batch_mean = torch.mean(y_true, dim=(1,2), keepdim=True)
            
            ss_res = torch.sum((y_true - y_pred)**2, dim=(1,2)).cpu().numpy()
            ss_tot = torch.sum((y_true - batch_mean)**2, dim=(1,2)).cpu().numpy()
            
            ss_res_list.extend(ss_res)
            ss_tot_list.extend(ss_tot)

    return np.concatenate(latents_list, axis=0), np.array(ss_res_list), np.array(ss_tot_list)

def bootstrap_clf_ci(y_true, y_probs, metric_func, n_bootstraps=1000):
    """Calculates 95% CI for classification metrics using bootstrapping."""
    scores = []
    rng = np.random.RandomState(42)
    indices = np.arange(len(y_true))
    
    for _ in range(n_bootstraps):
        sample_idx = resample(indices, random_state=rng)
        y_true_b, y_probs_b = y_true[sample_idx], y_probs[sample_idx]
        
        # Skip if sample lacks both classes (rare, but breaks AUC)
        if len(np.unique(y_true_b)) < 2:
            continue
            
        scores.append(metric_func(y_true_b, y_probs_b))
        
    return f"{np.mean(scores):.3f} [{np.percentile(scores, 2.5):.3f}, {np.percentile(scores, 97.5):.3f}]"

def bootstrap_recon_ci(ss_res_arr, ss_tot_arr, elements_per_sample, n_bootstraps=1000):
    """Calculates 95% CI for RMSE and R2 using per-sample sum of squares."""
    rmse_scores, r2_scores = [], []
    rng = np.random.RandomState(42)
    indices = np.arange(len(ss_res_arr))
    
    for _ in range(n_bootstraps):
        sample_idx = resample(indices, random_state=rng)
        b_ss_res = ss_res_arr[sample_idx]
        b_ss_tot = ss_tot_arr[sample_idx]
        
        total_res = np.sum(b_ss_res)
        total_tot = np.sum(b_ss_tot)
        total_elements = len(b_ss_res) * elements_per_sample
        
        rmse_scores.append(math.sqrt(total_res / total_elements))
        r2_scores.append(1.0 - (total_res / total_tot) if total_tot != 0 else 0.0)
        
    return (
        f"{np.mean(rmse_scores):.3f} [{np.percentile(rmse_scores, 2.5):.3f}, {np.percentile(rmse_scores, 97.5):.3f}]",
        f"{np.mean(r2_scores):.3f} [{np.percentile(r2_scores, 2.5):.3f}, {np.percentile(r2_scores, 97.5):.3f}]"
    )

def generate_umap_visualizations(latents, labels, h5_file_path, run_dir):
    print("\n--- Generating UMAP Visualizations ---")
    
    plot_dir = os.path.join(run_dir, "holdout_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    n_neighbors = 25
    min_dist = 0.01

    print(f"Calculating 2D UMAP (Neighbors: {n_neighbors}, Min Dist: {min_dist})...")
    reducer_2d = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
    umap_2d = reducer_2d.fit_transform(latents)

    print("Calculating 3D UMAP Projection...")
    reducer_3d = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=3, random_state=42)
    umap_3d = reducer_3d.fit_transform(latents)

    print("Extracting clinical text for hover tooltips...")
    df_gt_dict = {}
    with h5py.File(h5_file_path, 'r') as f:
        gt_group = f['GT']
        report_cols = [key for key in gt_group.keys() if key.startswith('report_')]
        for col in report_cols:
            df_gt_dict[col] = [val.decode('utf-8') for val in gt_group[col][:]]

    df_gt = pd.DataFrame(df_gt_dict)
    combined_reports = df_gt[report_cols].fillna('').astype(str).agg(' '.join, axis=1)
    clean_reports = combined_reports.str.strip().str.replace(r'\s+', ' ', regex=True)
    hover_snippets = clean_reports.str.slice(0, 250) + "..."

    label_strings = ['AFib' if val == 1 else 'Other' for val in labels]

    # --- Generate 2D PNG ---
    print("Saving 2D Plot...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=umap_2d[:, 0], y=umap_2d[:, 1], hue=label_strings,
        palette={'AFib': 'red', 'Other': 'lightgrey'},
        alpha=0.6, s=15, edgecolor=None
    )
    plt.title(f'Holdout 2D UMAP (n_neighbors={n_neighbors}): AFib', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "holdout_afib_umap_2d.png"), dpi=300)
    plt.close()

    # --- Generate 3D HTML ---
    print("Saving 3D Interactive Plot...")
    plot_df = pd.DataFrame({
        'UMAP_3D_1': umap_3d[:, 0], 'UMAP_3D_2': umap_3d[:, 1], 'UMAP_3D_3': umap_3d[:, 2],
        'Diagnosis': label_strings, 'Report_Snippet': hover_snippets
    }).sort_values(by='Diagnosis', ascending=False)

    fig_3d = px.scatter_3d(
        plot_df, x='UMAP_3D_1', y='UMAP_3D_2', z='UMAP_3D_3',
        color='Diagnosis', color_discrete_map={'AFib': 'red', 'Other': 'lightgrey'},
        opacity=0.6, hover_data={'Report_Snippet': True, 'UMAP_3D_1': False, 'UMAP_3D_2': False, 'UMAP_3D_3': False},
        title=f'Holdout Interactive 3D UMAP: AFib (Hover for Clinical Text)'
    )
    fig_3d.update_traces(marker=dict(size=3, line=dict(width=0)))
    fig_3d.update_layout(scene=dict(xaxis=dict(showbackground=False), yaxis=dict(showbackground=False), zaxis=dict(showbackground=False)))
    fig_3d.write_html(os.path.join(plot_dir, "holdout_afib_umap_3d_interactive.html"))
    
    print(f"Success! Visualizations saved to: {plot_dir}")

def generate_classification_curves(y_true, xgb_probs, lr_probs, run_dir):
    print("\n--- Generating ROC and PR Curves ---")
    plot_dir = os.path.join(run_dir, "holdout_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # ---------------------------------------------------------
    # ROC Curve Plot
    # ---------------------------------------------------------
    fpr_xgb, tpr_xgb, _ = roc_curve(y_true, xgb_probs)
    fpr_lr, tpr_lr, _ = roc_curve(y_true, lr_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_xgb, tpr_xgb, color='#1d3557', linewidth=2, label=f'XGBoost (AUC = {roc_auc_score(y_true, xgb_probs):.3f})')
    plt.plot(fpr_lr, tpr_lr, color='#e63946', linewidth=2, label=f'LogReg (AUC = {roc_auc_score(y_true, lr_probs):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title('ROC Curve (Holdout Evaluation)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(plot_dir, "holdout_roc_curve.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PR Curve Plot
    # ---------------------------------------------------------
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_true, xgb_probs)
    prec_lr, rec_lr, _ = precision_recall_curve(y_true, lr_probs)
    
    prevalence = np.mean(y_true)
    
    # Calculate F1 scores across all thresholds
    f1_xgb = np.divide(2 * (prec_xgb * rec_xgb), (prec_xgb + rec_xgb), 
                       out=np.zeros_like(prec_xgb), where=(prec_xgb + rec_xgb) != 0)
    f1_lr = np.divide(2 * (prec_lr * rec_lr), (prec_lr + rec_lr), 
                      out=np.zeros_like(prec_lr), where=(prec_lr + rec_lr) != 0)
    
    best_idx_xgb = np.argmax(f1_xgb)
    max_f1_xgb = f1_xgb[best_idx_xgb]
    best_rec_xgb = rec_xgb[best_idx_xgb]
    best_prec_xgb = prec_xgb[best_idx_xgb]
    
    best_idx_lr = np.argmax(f1_lr)
    max_f1_lr = f1_lr[best_idx_lr]
    best_rec_lr = rec_lr[best_idx_lr]
    best_prec_lr = prec_lr[best_idx_lr]
    
    plt.figure(figsize=(8, 6))
    plt.plot(rec_xgb, prec_xgb, color='#1d3557', linewidth=2, label=f'XGBoost (PR-AUC = {average_precision_score(y_true, xgb_probs):.3f})')
    plt.plot(rec_lr, prec_lr, color='#e63946', linewidth=2, label=f'LogReg (PR-AUC = {average_precision_score(y_true, lr_probs):.3f})')
    
    plt.plot(best_rec_xgb, best_prec_xgb, marker='o', markersize=9, color='#1d3557', markeredgecolor='white', 
             linestyle='None', label=f'XGB F1 ({max_f1_xgb:.3f})')
    plt.plot(best_rec_lr, best_prec_lr, marker='o', markersize=9, color='#e63946', markeredgecolor='white', 
             linestyle='None', label=f'LogReg F1 ({max_f1_lr:.3f})')
    
    plt.axhline(y=prevalence, color='gray', linestyle=':', linewidth=2, label=f'Prevalence Baseline ({prevalence:.3f})')
    
    plt.title('Precision-Recall Curve (Holdout Evaluation)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(os.path.join(plot_dir, "holdout_pr_curve.png"), dpi=300)
    plt.close()
    
    print(f"Saved holdout ROC and PR curves to {plot_dir}")

def generate_random_reconstructions(model, dataloader, run_dir):
    print("\n--- Generating Random Reconstructions Plot ---")
    plot_dir = os.path.join(run_dir, "holdout_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    model.eval()
    
    # Grab a single batch from the holdout dataloader
    x_batch = next(iter(dataloader))
    
    with torch.no_grad():
        reconstructed, _ = model(x_batch)
    
    # Permute to (Batch, Seq_Len, Channels) for easier plotting
    x_batch_np = x_batch.permute(0, 2, 1).cpu().numpy()
    reconstructed_np = reconstructed.permute(0, 2, 1).cpu().numpy()
    
    fig, axes = plt.subplots(5, 2, figsize=(20, 15))
    axes = axes.flatten()
    
    for i in range(min(10, len(x_batch_np))):
        # Plotting Channel 0 (Lead I)
        axes[i].plot(x_batch_np[i, :, 0], label="Original (Lead I)", alpha=0.7)
        axes[i].plot(reconstructed_np[i, :, 0], label="Reconstruction", color='red', linestyle='--')
        axes[i].set_title(f"Holdout Random ECG Sample {i+1}")
        axes[i].legend(loc='upper right')
        
    plt.tight_layout()
    save_path = os.path.join(plot_dir, "holdout_random_reconstructions.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Saved random reconstructions to {save_path}")

# ================================
# 4. Main Execution
# ================================
def main():
    print("\n" + "="*60)
    print(" HOLDOUT EVALUATION PIPELINE ")
    print("="*60)

    # 1. Load config
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    print(f"Loaded config for Latent Dim: {config['latent_dim']}, Base Filters: {config['base_filters']}")

    # 2. Initialize Model
    model = ConvAutoencoder(
        seq_len=config['seq_len'], in_channels=8, latent_dim=config['latent_dim'], 
        base_filters=config['base_filters'], kernel_size=config['kernel_size'],
        num_layers=config['num_layers'], pool_size=config['pool_size'], 
        activation=config['activation'], dropout_rate=config['dropout_rate'], 
        norm_type=config['norm_type'], pooling_type=config['pooling_type'], masking_ratio=0.0
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    print("Loaded Best Model Weights.")

    # 3. Process Training Data for Classifier
    print("\n--- Processing Training Set (for Classifier Fitting) ---")
    train_dataset = ECGDataset(TRAIN_DATA_PATH, config['seq_len'])
    train_loader = FastTensorDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
    y_train = extract_afib_labels(TRAIN_DATA_PATH)
    
    X_train, _ = get_latents_and_reconstruction(model, train_loader)
    
    # Clean up RAM
    del train_dataset, train_loader
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Train Final Classifiers
    print("Training Final Classifiers on Full Training Latents...")
    num_pos = sum(y_train)
    scale_pos_weight = (len(y_train) - num_pos) / num_pos if num_pos > 0 else 1.0
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic', scale_pos_weight=scale_pos_weight,
        tree_method='hist', n_estimators=150, learning_rate=0.05, max_depth=5,
        eval_metric='auc', random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    lr_model = LogisticRegression(class_weight='balanced', max_iter=1500, random_state=42)
    lr_model.fit(X_train, y_train)

    # 5. Process Holdout Data
    print("\n--- Processing Holdout Set ---")
    holdout_dataset = ECGDataset(HOLDOUT_DATA_PATH, config['seq_len'])
    holdout_loader = FastTensorDataLoader(holdout_dataset, batch_size=config['batch_size'], shuffle=False)
    y_holdout = extract_afib_labels(HOLDOUT_DATA_PATH)
    
    # Get arrays
    X_holdout, holdout_ss_res, holdout_ss_tot = get_latents_and_reconstruction(model, holdout_loader)

    # 6. Bootstrap Evaluations
    print("Calculating 95% Confidence Intervals via Bootstrapping (1000 iterations)...")
    
    # Reconstruction Bootstrapping
    elements_per_sample = config['seq_len'] * 8 # 8 channels
    rmse_str, r2_str = bootstrap_recon_ci(holdout_ss_res, holdout_ss_tot, elements_per_sample)

    # Classification Bootstrapping
    xgb_probs = xgb_model.predict_proba(X_holdout)[:, 1]
    lr_probs = lr_model.predict_proba(X_holdout)[:, 1]

    xgb_auc_str = bootstrap_clf_ci(y_holdout, xgb_probs, roc_auc_score)
    xgb_prauc_str = bootstrap_clf_ci(y_holdout, xgb_probs, average_precision_score)
    lr_auc_str = bootstrap_clf_ci(y_holdout, lr_probs, roc_auc_score)
    lr_prauc_str = bootstrap_clf_ci(y_holdout, lr_probs, average_precision_score)

    # 7. Print Final Results
    print("\n" + "="*60)
    print(" HOLDOUT EVALUATION RESULTS (WITH 95% CIs) ")
    print("="*60)
    print("Reconstruction Metrics:")
    print(f"  RMSE: {rmse_str}")
    print(f"  R2:   {r2_str}")
    print("\nClassification Metrics:")
    print(f"  XGBoost AUC:    {xgb_auc_str}")
    print(f"  XGBoost PR-AUC: {xgb_prauc_str}")
    print(f"  LogReg AUC:     {lr_auc_str}")
    print(f"  LogReg PR-AUC:  {lr_prauc_str}")

    print("\nSAVING RANDOM RECONSTRUCTIONS PLOT...")
    generate_random_reconstructions(model, holdout_loader, RUN_DIR)
    
    print("\nSAVING UMAP VISUALIZATIONS...")
    np.save(os.path.join(RUN_DIR, "holdout_latents.npy"), X_holdout)
    np.save(os.path.join(RUN_DIR, "holdout_labels.npy"), y_holdout)
    generate_umap_visualizations(X_holdout, y_holdout, HOLDOUT_DATA_PATH, RUN_DIR)

    print("\nSAVING CLASSIFICATION CURVES...")
    generate_classification_curves(y_holdout, xgb_probs, lr_probs, RUN_DIR)

if __name__ == "__main__":
    main()
    print("="*60)
    

if __name__ == "__main__":
    main()