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
CSV_PATH = os.path.join(BASE_DIR, "data/unzipped/MIMIC_IV_ECG_CSV_MICROVOLTS_v3/record_list.csv")

RUN_DIRS = [
    os.path.join(BASE_DIR, "model_development/experiments/GridRun_001_2804_1735"),
    os.path.join(BASE_DIR, "model_development/experiments/GridRun_001_2904_1242"),
    os.path.join(BASE_DIR, "model_development/experiments/GridRun_003_2204_1304"),
    os.path.join(BASE_DIR, "model_development/experiments/GridRun_002_2204_0913"),
    os.path.join(BASE_DIR, "model_development/experiments/GridRun_001_2804_1029")
]

EXACT_TARGETS = ["ATRIAL FIBRILLATION", "Atrial fibrillation", "Atrial fibrillation."]

# ================================
# 2. Classes (Optimized for RAM)
# ================================
class BaseECGMemoryHolder:
    """Holds the full-length ECG in memory to avoid repetitive disk reads."""
    def __init__(self, h5_file_path):
        print(f"Loading raw dataset into RAM from {h5_file_path}...")
        with h5py.File(h5_file_path, 'r') as h5f:
            self.raw_data = torch.tensor(h5f['rhythm_filtered'][:], dtype=torch.float32).permute(0, 2, 1)

class FastTensorDataLoader:
    """Slices and standardizes data dynamically on the GPU based on the run seq_len."""
    def __init__(self, base_holder, batch_size, seq_len, shuffle=False):
        self.raw_data = base_holder.raw_data
        self.seq_len = seq_len
        self.indices = torch.arange(len(self.raw_data), dtype=torch.long)
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
        
        # 1. Slice to current model's seq_len and move to GPU
        x_batch = self.raw_data[batch_idx, :, :self.seq_len].to(DEVICE)
        
        # 2. Standardize On-the-Fly (Per-sample, per-channel along the temporal axis)
        means = x_batch.mean(dim=2, keepdim=True)
        stds = x_batch.std(dim=2, keepdim=True)
        x_batch = (x_batch - means) / (stds + 1e-8)
        
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
def get_safe_holdout_mask(train_h5_path, holdout_h5_path, csv_path):
    print("Generating pure holdout subset mask via record_list.csv...")
    df_records = pd.read_csv(csv_path)
    study_to_subject = dict(zip(df_records['study_id'].astype(str), df_records['subject_id'].astype(str)))
    
    with h5py.File(train_h5_path, 'r') as f_train, h5py.File(holdout_h5_path, 'r') as f_holdout:
        train_studies = [val.decode('utf-8') if isinstance(val, bytes) else str(val) for val in f_train['GT']['study_id'][:]]
        holdout_studies = [val.decode('utf-8') if isinstance(val, bytes) else str(val) for val in f_holdout['GT']['study_id'][:]]
        
    train_patients = set([study_to_subject.get(s, f"UNMAPPED_{s}") for s in train_studies])
    holdout_patients = [study_to_subject.get(s, f"UNMAPPED_{s}") for s in holdout_studies]
    
    safe_mask = np.array([p not in train_patients for p in holdout_patients])
    print(f"Mask generation complete. Found {np.sum(safe_mask)} clean ECGs out of {len(safe_mask)} total.")
    return safe_mask

def extract_afib_labels(h5_file_path):
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
    model.eval()
    latents_list, ss_res_list, ss_tot_list = [], [], []
    
    with torch.no_grad():
        for x_batch in dataloader:
            outputs, latents = model(x_batch)
            latents_list.append(latents.cpu().numpy())
            
            y_true, y_pred = x_batch, outputs
            batch_mean = torch.mean(y_true, dim=(1,2), keepdim=True)
            
            ss_res = torch.sum((y_true - y_pred)**2, dim=(1,2)).cpu().numpy()
            ss_tot = torch.sum((y_true - batch_mean)**2, dim=(1,2)).cpu().numpy()
            
            ss_res_list.extend(ss_res)
            ss_tot_list.extend(ss_tot)

    return np.concatenate(latents_list, axis=0), np.array(ss_res_list), np.array(ss_tot_list)

def bootstrap_clf_ci(y_true, y_probs, metric_func, baseline_probs=None, n_bootstraps=1000):
    base_score = metric_func(y_true, y_probs)
    
    scores = []
    diffs = []
    rng = np.random.RandomState(42)
    indices = np.arange(len(y_true))
    
    for _ in range(n_bootstraps):
        sample_idx = resample(indices, random_state=rng)
        y_true_b, y_probs_b = y_true[sample_idx], y_probs[sample_idx]
        
        if len(np.unique(y_true_b)) < 2:
            continue
            
        score_b = metric_func(y_true_b, y_probs_b)
        scores.append(score_b)
        
        if baseline_probs is not None:
            baseline_probs_b = baseline_probs[sample_idx]
            baseline_score_b = metric_func(y_true_b, baseline_probs_b)
            diffs.append(score_b - baseline_score_b)
            
    ci_str = f"{base_score:.3f} [{np.percentile(scores, 2.5):.3f}, {np.percentile(scores, 97.5):.3f}]"
    
    if baseline_probs is not None and len(diffs) > 0:
        diffs = np.array(diffs)
        # Calculate two-tailed empirical P-value
        p_val = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
        # Handle the edge case where it never crosses zero in 1000 bootstraps
        p_str = f"p < 0.001" if p_val == 0 else f"p = {p_val:.3f}"
        return f"{ci_str} ({p_str})"
        
    return ci_str

def bootstrap_recon_ci(ss_res_arr, ss_tot_arr, elements_per_sample, baseline_res=None, baseline_tot=None, n_bootstraps=1000):
    base_total_res = np.sum(ss_res_arr)
    base_total_tot = np.sum(ss_tot_arr)
    base_total_elements = len(ss_res_arr) * elements_per_sample
    
    base_rmse = math.sqrt(base_total_res / base_total_elements)
    base_r2 = 1.0 - (base_total_res / base_total_tot) if base_total_tot != 0 else 0.0

    rmse_scores, r2_scores = [], []
    rmse_diffs, r2_diffs = [], []
    
    rng = np.random.RandomState(42)
    indices = np.arange(len(ss_res_arr))
    
    for _ in range(n_bootstraps):
        sample_idx = resample(indices, random_state=rng)
        b_ss_res, b_ss_tot = ss_res_arr[sample_idx], ss_tot_arr[sample_idx]
        
        total_res = np.sum(b_ss_res)
        total_tot = np.sum(b_ss_tot)
        total_elements = len(b_ss_res) * elements_per_sample
        
        rmse_b = math.sqrt(total_res / total_elements)
        r2_b = 1.0 - (total_res / total_tot) if total_tot != 0 else 0.0
        rmse_scores.append(rmse_b)
        r2_scores.append(r2_b)
        
        if baseline_res is not None and baseline_tot is not None:
            b_base_res, b_base_tot = baseline_res[sample_idx], baseline_tot[sample_idx]
            base_total_res_b = np.sum(b_base_res)
            base_total_tot_b = np.sum(b_base_tot)
            
            base_rmse_b = math.sqrt(base_total_res_b / total_elements)
            base_r2_b = 1.0 - (base_total_res_b / base_total_tot_b) if base_total_tot_b != 0 else 0.0
            
            rmse_diffs.append(rmse_b - base_rmse_b)
            r2_diffs.append(r2_b - base_r2_b)

    rmse_ci = f"{base_rmse:.3f} [{np.percentile(rmse_scores, 2.5):.3f}, {np.percentile(rmse_scores, 97.5):.3f}]"
    r2_ci = f"{base_r2:.3f} [{np.percentile(r2_scores, 2.5):.3f}, {np.percentile(r2_scores, 97.5):.3f}]"

    if baseline_res is not None and len(rmse_diffs) > 0:
        rmse_diffs, r2_diffs = np.array(rmse_diffs), np.array(r2_diffs)
        
        p_rmse = 2 * min(np.mean(rmse_diffs <= 0), np.mean(rmse_diffs >= 0))
        p_r2 = 2 * min(np.mean(r2_diffs <= 0), np.mean(r2_diffs >= 0))
        
        str_p_rmse = f"p < 0.001" if p_rmse == 0 else f"p = {p_rmse:.3f}"
        str_p_r2 = f"p < 0.001" if p_r2 == 0 else f"p = {p_r2:.3f}"
        
        return f"{rmse_ci} ({str_p_rmse})", f"{r2_ci} ({str_p_r2})"

    return rmse_ci, r2_ci

def generate_umap_visualizations(latents, labels, h5_file_path, run_dir, file_suffix, subset_mask=None, max_points=5000):
    print(f"\n--- Generating UMAP Visualizations ({file_suffix}) ---")
    plot_dir = os.path.join(run_dir, "holdout_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # --- 1. DOWNSAMPLING ---
    total_points = len(latents)
    if total_points > max_points:
        print(f"Downsampling from {total_points} to {max_points} points for cleaner visualization...")
        np.random.seed(42)
        # We need the indices to map back to the original dataframe for hover text later
        original_indices = np.arange(total_points)
        sample_idx = np.random.choice(total_points, max_points, replace=False)
        
        latents_viz = latents[sample_idx]
        labels_viz = labels[sample_idx]
        kept_indices = original_indices[sample_idx]
    else:
        latents_viz = latents
        labels_viz = labels
        kept_indices = np.arange(total_points)

    # --- 2. UMAP COMPUTATION ---
    n_neighbors = 25
    min_dist = 0.01

    print("Calculating 2D and 3D UMAP projections...")
    reducer_2d = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
    umap_2d = reducer_2d.fit_transform(latents_viz)

    reducer_3d = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=3, random_state=42)
    umap_3d = reducer_3d.fit_transform(latents_viz)

    # --- 3. CLINICAL TEXT EXTRACTION ---
    df_gt_dict = {}
    with h5py.File(h5_file_path, 'r') as f:
        gt_group = f['GT']
        report_cols = [key for key in gt_group.keys() if key.startswith('report_')]
        for col in report_cols:
            df_gt_dict[col] = [val.decode('utf-8') for val in gt_group[col][:]]

    df_gt = pd.DataFrame(df_gt_dict)
    
    if subset_mask is not None:
        df_gt = df_gt.iloc[subset_mask].reset_index(drop=True)
        
    df_gt_viz = df_gt.iloc[kept_indices].copy()
        
    combined_reports = df_gt_viz[report_cols].fillna('').astype(str).agg(' '.join, axis=1)
    clean_reports = combined_reports.str.strip().str.replace(r'\s+', ' ', regex=True)
    hover_snippets = clean_reports.str.slice(0, 250) + "..."
    hover_snippets = hover_snippets.reset_index(drop=True) 

    label_strings = ['AFib' if val == 1 else 'Other' for val in labels_viz]

    # --- 4. PREPARE DATAFRAME FOR Z-ORDER SORTING ---
    plot_df = pd.DataFrame({
        'UMAP_2D_1': umap_2d[:, 0],
        'UMAP_2D_2': umap_2d[:, 1],
        'UMAP_3D_1': umap_3d[:, 0],
        'UMAP_3D_2': umap_3d[:, 1],
        'UMAP_3D_3': umap_3d[:, 2],
        'Diagnosis': label_strings,
        'Report_Snippet': hover_snippets
    })

    # Sort so 'Other' is at the top of the dataframe (drawn first), and 'AFib' is at the bottom (drawn last)
    plot_df = plot_df.sort_values(by='Diagnosis', ascending=False)

    # --- 5. GENERATE 2D PNG ---
    print("Saving 2D Scatter Plot...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=plot_df,
        x='UMAP_2D_1', 
        y='UMAP_2D_2', 
        hue='Diagnosis',
        palette={'AFib': 'red', 'Other': 'lightgrey'},
        alpha=0.6, 
        s=15, 
        edgecolor=None
    )
    plt.title(f'Holdout 2D UMAP ({file_suffix}): AFib (Max {max_points} pts)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"holdout_afib_umap_2d_{file_suffix}.png"), dpi=300)
    plt.close()

    # --- 6. GENERATE 3D HTML ---
    print("Saving 3D Interactive Plot...")
    fig_3d = px.scatter_3d(
        plot_df, 
        x='UMAP_3D_1', 
        y='UMAP_3D_2', 
        z='UMAP_3D_3',
        color='Diagnosis', 
        color_discrete_map={'AFib': 'red', 'Other': 'lightgrey'},
        opacity=0.6, 
        hover_data={'Report_Snippet': True, 'UMAP_3D_1': False, 'UMAP_3D_2': False, 'UMAP_3D_3': False},
        title=f'Holdout Interactive 3D UMAP ({file_suffix}) (Max {max_points} pts)'
    )
    fig_3d.update_traces(marker=dict(size=3, line=dict(width=0)))
    fig_3d.update_layout(scene=dict(xaxis=dict(showbackground=False), yaxis=dict(showbackground=False), zaxis=dict(showbackground=False)))
    fig_3d.write_html(os.path.join(plot_dir, f"holdout_afib_umap_3d_{file_suffix}.html"))

def generate_classification_curves(y_true, xgb_probs, lr_probs, run_dir, file_suffix):
    print(f"\n--- Generating ROC and PR Curves ({file_suffix}) ---")
    plot_dir = os.path.join(run_dir, "holdout_plots")
    os.makedirs(plot_dir, exist_ok=True)

    fpr_xgb, tpr_xgb, _ = roc_curve(y_true, xgb_probs)
    fpr_lr, tpr_lr, _ = roc_curve(y_true, lr_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_xgb, tpr_xgb, color='#1d3557', linewidth=2, label=f'XGBoost (AUC = {roc_auc_score(y_true, xgb_probs):.3f})')
    plt.plot(fpr_lr, tpr_lr, color='#e63946', linewidth=2, label=f'LogReg (AUC = {roc_auc_score(y_true, lr_probs):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title(f'ROC Curve (C1)', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc='lower right', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(plot_dir, f"holdout_roc_curve_{file_suffix}.png"), dpi=300)
    plt.close()

    prec_xgb, rec_xgb, _ = precision_recall_curve(y_true, xgb_probs)
    prec_lr, rec_lr, _ = precision_recall_curve(y_true, lr_probs)
    
    prevalence = np.mean(y_true)
    
    f1_xgb = np.divide(2 * (prec_xgb * rec_xgb), (prec_xgb + rec_xgb), out=np.zeros_like(prec_xgb), where=(prec_xgb + rec_xgb) != 0)
    f1_lr = np.divide(2 * (prec_lr * rec_lr), (prec_lr + rec_lr), out=np.zeros_like(prec_lr), where=(prec_lr + rec_lr) != 0)
    
    best_idx_xgb = np.argmax(f1_xgb)
    best_idx_lr = np.argmax(f1_lr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(rec_xgb, prec_xgb, color='#1d3557', linewidth=2, label=f'XGBoost (PR-AUC = {average_precision_score(y_true, xgb_probs):.3f})')
    plt.plot(rec_lr, prec_lr, color='#e63946', linewidth=2, label=f'LogReg (PR-AUC = {average_precision_score(y_true, lr_probs):.3f})')
    
    plt.plot(rec_xgb[best_idx_xgb], prec_xgb[best_idx_xgb], marker='o', markersize=9, color='#1d3557', markeredgecolor='white', linestyle='None', label=f'XGB F1 ({f1_xgb[best_idx_xgb]:.3f})')
    plt.plot(rec_lr[best_idx_lr], prec_lr[best_idx_lr], marker='o', markersize=9, color='#e63946', markeredgecolor='white', linestyle='None', label=f'LogReg F1 ({f1_lr[best_idx_lr]:.3f})')
    
    plt.axhline(y=prevalence, color='gray', linestyle=':', linewidth=2, label=f'Prevalence Baseline ({prevalence:.3f})')
    
    plt.title(f'Precision-Recall Curve (C1)', fontsize=16)
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.legend(loc='upper right', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(plot_dir, f"holdout_pr_curve_{file_suffix}.png"), dpi=300)
    plt.close()

def generate_random_reconstructions(model, raw_data, seq_len, run_dir, file_suffix, subset_mask=None, num_samples=10):
    print(f"\n--- Generating Random Reconstructions Plot ({file_suffix}) ---")
    plot_dir = os.path.join(run_dir, "holdout_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    model.eval()
    
    # Determine which indices are allowed to sample from
    if subset_mask is not None:
        valid_indices = np.where(subset_mask)[0]
    else:
        valid_indices = np.arange(len(raw_data))
        
    # Pick random indices from the pool
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(valid_indices, size=min(num_samples, len(valid_indices)), replace=False)
    
    # Extract data for these indices, slice to seq_len, and move to GPU
    x_batch = raw_data[sample_indices, :, :seq_len].to(DEVICE)
    
    # Standardize identically to the FastTensorDataLoader
    means = x_batch.mean(dim=2, keepdim=True)
    stds = x_batch.std(dim=2, keepdim=True)
    x_batch = (x_batch - means) / (stds + 1e-8)
    
    with torch.no_grad():
        reconstructed, _ = model(x_batch)
    
    # Permute to (Batch, Seq_Len, Channels) for easier plotting
    x_batch_np = x_batch.permute(0, 2, 1).cpu().numpy()
    reconstructed_np = reconstructed.permute(0, 2, 1).cpu().numpy()
    
    fig, axes = plt.subplots(5, 2, figsize=(20, 15))
    axes = axes.flatten()
    
    for i in range(len(sample_indices)):
        # Plotting Channel 0 (Lead I)
        axes[i].plot(x_batch_np[i, :, 0], label="Original (Lead I)", alpha=0.7)
        axes[i].plot(reconstructed_np[i, :, 0], label="Reconstruction", color='red', linestyle='--')
        axes[i].set_title(f"Holdout ECG ({file_suffix}) - Original Index: {sample_indices[i]}")
        axes[i].legend(loc='upper right')
        
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f"holdout_random_reconstructions_{file_suffix}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Saved: {save_path}")

# ================================
# 4. Evaluation Module
# ================================
def evaluate_subset(X_subset, y_subset, ss_res_subset, ss_tot_subset, xgb_model, lr_model, elements_per_sample, run_dir, file_suffix, subset_mask=None, baseline_dict=None):
    print(f"\nEvaluating subset: {file_suffix.upper()} (N={len(y_subset)})")
    
    xgb_probs = xgb_model.predict_proba(X_subset)[:, 1]
    lr_probs = lr_model.predict_proba(X_subset)[:, 1]

    # Extract baselines if they exist
    b_xgb = baseline_dict['xgb_probs'] if baseline_dict else None
    b_lr = baseline_dict['lr_probs'] if baseline_dict else None
    b_res = baseline_dict['ss_res'] if baseline_dict else None
    b_tot = baseline_dict['ss_tot'] if baseline_dict else None

    rmse_str, r2_str = bootstrap_recon_ci(ss_res_subset, ss_tot_subset, elements_per_sample, b_res, b_tot)
    xgb_auc_str = bootstrap_clf_ci(y_subset, xgb_probs, roc_auc_score, b_xgb)
    xgb_prauc_str = bootstrap_clf_ci(y_subset, xgb_probs, average_precision_score, b_xgb)
    lr_auc_str = bootstrap_clf_ci(y_subset, lr_probs, roc_auc_score, b_lr)
    lr_prauc_str = bootstrap_clf_ci(y_subset, lr_probs, average_precision_score, b_lr)

    print(f"--- RESULTS: {file_suffix.upper()} ---")
    print(f"  RMSE:           {rmse_str}")
    print(f"  R2:             {r2_str}")
    print(f"  XGBoost AUC:    {xgb_auc_str}")
    print(f"  XGBoost PR-AUC: {xgb_prauc_str}")
    print(f"  LogReg AUC:     {lr_auc_str}")
    print(f"  LogReg PR-AUC:  {lr_prauc_str}")

    generate_umap_visualizations(X_subset, y_subset, HOLDOUT_DATA_PATH, run_dir, file_suffix, subset_mask)
    generate_classification_curves(y_subset, xgb_probs, lr_probs, run_dir, file_suffix)

# ================================
# 5. Main Execution
# ================================
def main():
    print("\n" + "="*60)
    print(" BATCH HOLDOUT EVALUATION PIPELINE ")
    print("="*60)

    if not RUN_DIRS:
        print("No run directories specified. Exiting.")
        return

    safe_mask = get_safe_holdout_mask(TRAIN_DATA_PATH, HOLDOUT_DATA_PATH, CSV_PATH)

    print("\n--- Loading Datasets into RAM ---")
    train_base = BaseECGMemoryHolder(TRAIN_DATA_PATH)
    holdout_base = BaseECGMemoryHolder(HOLDOUT_DATA_PATH)
    
    print("Extracting Clinical Labels...")
    y_train = extract_afib_labels(TRAIN_DATA_PATH)
    y_holdout = extract_afib_labels(HOLDOUT_DATA_PATH)

    baseline_cache = {
        "full_holdout": None,
        "clean_subset": None
    }
    
    is_first_model = True

    for run_dir in RUN_DIRS:
        print("\n\n" + "#"*60)
        print(f" EVALUATING RUN: {os.path.basename(run_dir)}")
        print("#"*60)
        
        config_path = os.path.join(run_dir, "config.json")
        model_weights_path = os.path.join(run_dir, "best_fold_model.pth")
        
        if not os.path.exists(config_path) or not os.path.exists(model_weights_path):
            print(f"Skipping {run_dir} - Missing config or weights.")
            continue

        with open(config_path, "r") as f:
            config = json.load(f)

        current_seq_len = config.get('seq_len', 5000)

        model = ConvAutoencoder(
            seq_len=current_seq_len, in_channels=8, latent_dim=config['latent_dim'], 
            base_filters=config['base_filters'], kernel_size=config['kernel_size'],
            num_layers=config['num_layers'], pool_size=config['pool_size'], 
            activation=config['activation'], dropout_rate=config['dropout_rate'], 
            norm_type=config['norm_type'], pooling_type=config['pooling_type'], masking_ratio=0.0
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))

        print(f"\n--- Processing Training Set (seq_len={current_seq_len}) ---")
        train_loader = FastTensorDataLoader(train_base, batch_size=config['batch_size'], seq_len=current_seq_len, shuffle=False)
        X_train, _, _ = get_latents_and_reconstruction(model, train_loader)
        
        print("Training Final Classifiers...")
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

        print("\n--- Processing Holdout Set ---")
        holdout_loader = FastTensorDataLoader(holdout_base, batch_size=config['batch_size'], seq_len=current_seq_len, shuffle=False)
        
        # Calculate latents for the entire holdout set
        X_holdout, holdout_ss_res, holdout_ss_tot = get_latents_and_reconstruction(model, holdout_loader)
        
        elements_per_sample = current_seq_len * 8

        # --- EVALUATION 1: Full Holdout ---
        xgb_probs_full = xgb_model.predict_proba(X_holdout)[:, 1]
        lr_probs_full = lr_model.predict_proba(X_holdout)[:, 1]
        
        if is_first_model:
            baseline_cache["full_holdout"] = {
                'xgb_probs': xgb_probs_full, 'lr_probs': lr_probs_full,
                'ss_res': holdout_ss_res, 'ss_tot': holdout_ss_tot
            }
            
        generate_random_reconstructions(model, holdout_base.raw_data, current_seq_len, run_dir, "full_holdout", subset_mask=None)
        evaluate_subset(X_holdout, y_holdout, holdout_ss_res, holdout_ss_tot, 
                        xgb_model, lr_model, elements_per_sample, run_dir, "full_holdout", 
                        baseline_dict=None if is_first_model else baseline_cache["full_holdout"])
                        
        # --- EVALUATION 2: Clean Subset ---
        X_clean = X_holdout[safe_mask]
        y_clean = y_holdout[safe_mask]
        ss_res_clean = holdout_ss_res[safe_mask]
        ss_tot_clean = holdout_ss_tot[safe_mask]
        
        xgb_probs_clean = xgb_model.predict_proba(X_clean)[:, 1]
        lr_probs_clean = lr_model.predict_proba(X_clean)[:, 1]

        if is_first_model:
            baseline_cache["clean_subset"] = {
                'xgb_probs': xgb_probs_clean, 'lr_probs': lr_probs_clean,
                'ss_res': ss_res_clean, 'ss_tot': ss_tot_clean
            }
        
        generate_random_reconstructions(model, holdout_base.raw_data, current_seq_len, run_dir, "clean_subset", subset_mask=safe_mask)
        evaluate_subset(X_clean, y_clean, ss_res_clean, ss_tot_clean, 
                        xgb_model, lr_model, elements_per_sample, run_dir, "clean_subset", subset_mask=safe_mask,
                        baseline_dict=None if is_first_model else baseline_cache["clean_subset"])
                        
        is_first_model = False

if __name__ == "__main__":
    main()