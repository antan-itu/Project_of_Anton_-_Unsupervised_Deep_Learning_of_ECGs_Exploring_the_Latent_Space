"""
Filename: autoencoder_training_script.py
Author: antan
Date: 2026-06-01
Version: 9.0
Description: This script trains a CNN-based autoencoder and subsequently runs XGBoost and logistic regression for AF detection.
"""
### Importing libraries and setting seeds for reproducibility ###
import os
import random
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
import h5py
import math
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import scipy.stats as st
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

# Setting seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

# Detecting GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

### Grid search parameters and directories ###
BASE_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space"
OUTPUT_DIR = os.path.join(BASE_DIR, "model_development/experiments")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data/MIMIC_IV_ECG_HDF5/mimic_iv_train.h5")
os.makedirs(OUTPUT_DIR, exist_ok=True) # Make sure the directory exists

# Defining sequence length, number of channels (leads), and folds for cross-validation 
SEQ_LEN = 5000  # The length of the ECG recording 500Hz and 10 seconds
IN_CHANNELS = 8 # The number of ECG leads 
K_FOLDS = 5

### Setting the hyperparameter for the grid search ##
GRID = {
  'batch_size': [128],
  'pooling_type': ['average'],
  'latent_dim': [128],
  'learning_rate': [0.0005],
  'base_filters': [128],
  'kernel_size': [75],
  'num_layers': [3], 
  'pool_size': [10], 
  'activation': ['relu'],
  'norm_type': ['layer'],
  'dropout_rate': [0],
  'loss_func': ['huber'],
  'masking_ratio': [0]
}

keys, values = zip(*GRID.items())
EXPERIMENT_COMBINATIONS = [dict(zip(keys, v)) for v in itertools.product(*values)]

### The class loads the dataset into RAM and standardizes it ###
class MIMIC:
  def __init__(self, h5_file_path, seq_len):
      print(f"\nLoading dataset {h5_file_path}...")
      with h5py.File(h5_file_path, 'r') as h5f:
          # Slice sequence to the defined length
          self.data = torch.tensor(h5f['rhythm_filtered'][:], dtype=torch.float32).permute(0, 2, 1)[:, :, :seq_len]
      print(f"Dataset loaded - Shape: {self.data.shape}")
      # Z-score normalization per sample
      print("Normalizing data...")
      means = self.data.mean(dim=2, keepdim=True)
      stds = self.data.std(dim=2, keepdim=True)
      self.data -= means
      self.data /= (stds + 1e-8)
      del means, stds
      gc.collect()
      print("Data normalized")

### The custom data loader reduces multiprocessing overhead ###
class DataLoader:
    # Uses indices to load batches from the dataset
    def __init__(self, dataset, indices, batch_size, shuffle=False):
        self.dataset = dataset
        self.indices = torch.tensor(indices, dtype=torch.long)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = math.ceil(len(self.indices) / self.batch_size)
        
    # Epoch initialization
    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(len(self.indices))
            self.indices = self.indices[perm]
        self.current_batch = 0
        return self
        
    # Fetching the next batch
    def __next__(self):
        if self.current_batch >= self.n_batches:
            raise StopIteration
        
        start = self.current_batch * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        batch_idx = self.indices[start:end]
        
        x_batch = self.dataset.data[batch_idx].to(DEVICE)
        self.current_batch += 1
        
        return x_batch, x_batch 
        
    # Returns total number of batches
    def __len__(self):
        return self.n_batches

### Early stopping class - checks validation loss and automatically stop training if no improvement after certain epochs (patience) ###
class EarlyStopping:
  def __init__(self, patience=5, delta=0.001):
      self.patience = patience
      self.delta = delta
      self.counter = 0
      self.best_loss = None
      self.early_stop = False
      self.best_model_state = None

  def __call__(self, val_loss, model):
      if math.isnan(val_loss):
          self.early_stop = True
          return
      if self.best_loss is None:
          self.best_loss = val_loss
          self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
      elif val_loss > self.best_loss - self.delta:
          self.counter += 1
          if self.counter >= self.patience:
              self.early_stop = True
      else:
          self.best_loss = val_loss
          self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
          self.counter = 0

### The autoencoder architecture ###
class Autoencoder(nn.Module):
  def __init__(self, seq_len, in_channels, latent_dim, base_filters, kernel_size, # Takes the hyperparameters as input
               num_layers, pool_size, activation, dropout_rate, norm_type, pooling_type, masking_ratio=0.0):
      super(Autoencoder, self).__init__()
      self.seq_len = seq_len # 
      self.masking_ratio = masking_ratio 
      padding = kernel_size // 2 # Uses padding to add zeros on both sides, so the output has the same length as the input after convolution
      encoder_layers = []
      current_channels = in_channels
      
      # For each layer, the number of filters doubles, and the stride or pooling is applied based on the pooling type
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
          
      self.encoder = nn.Sequential(*encoder_layers) #
      
      # Creating a dummy input to determine the shape of the encoders output - it's needed for the fully connected layers
      dummy_input = torch.zeros(1, in_channels, seq_len)
      dummy_output = self.encoder(dummy_input)
      self.shape_before_flatten = dummy_output.shape[1:]
      flattened_size = int(np.prod(self.shape_before_flatten))
      self.fc_latent = nn.Linear(flattened_size, latent_dim)
      self.fc_decoder_input = nn.Linear(latent_dim, flattened_size)
      
      decoder_layers = []
      # The activation function is selected
      if activation == 'leaky_relu': 
          decoder_layers.append(nn.LeakyReLU())
      else:
          decoder_layers.append(nn.ReLU())
    
      # For each layer in the decoder, the number of filters halves, and the stride or upsampling is applied based on the pooling type
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
      self.final_conv = nn.Conv1d(in_channels, in_channels, 1) # Input and output channels are the same, by using a final 1x1 convolution

  # Adding masking
  def forward(self, x):
      if self.training and self.masking_ratio > 0.0:
          mask = (torch.rand_like(x) > self.masking_ratio).float()
          x_input = x * mask
      else:
          x_input = x

      encoded = self.encoder(x_input)                                                  # The input goes through the encoder
      flattened = encoded.view(encoded.size(0), -1)                                    # It's then flattened
      latent = self.fc_latent(flattened)                                               # And passed through a fully connected layer - the latent space is created
      decoded_input = self.fc_decoder_input(latent)                                    # The latents space is passed through another fully connected layer
      reshaped = decoded_input.view(decoded_input.size(0), *self.shape_before_flatten) # Reshaping the output to fit the decoder's expected input shape
      decoded = self.decoder(reshaped)                                                 # The reshaped data goes through the decoder
      
      # The output of the decoder is trimmed or padded to ensure a sequence length that matches the original input
      if decoded.size(2) > self.seq_len:
          decoded = decoded[:, :, :self.seq_len]
      elif decoded.size(2) < self.seq_len:
          pad_size = self.seq_len - decoded.size(2)
          decoded = torch.nn.functional.pad(decoded, (0, pad_size))
          
      out = self.final_conv(decoded) # Ensuring the output has the same number of channels as the input
      return out, latent # Returning reconstructed output and the latent space

### Helper functions for evaluation and plotting ###
# This function calculates the 95% confidence interval for the different metrics
def confidence_intervals(metric_list):
    n_folds = len(metric_list) # Number of folds in the cross-validation
    mean_val = np.mean(metric_list) # The average value of the metric across all folds
    std_err = np.std(metric_list, ddof=1) / np.sqrt(n_folds) # Standard error of the mean
    margin_of_error = st.t.ppf(1 - 0.025, n_folds - 1) * std_err if n_folds > 1 else 0 # Margin of error for the 95% confidence interval
    return f"[{round(mean_val - margin_of_error, 3):.3f}, {round(mean_val + margin_of_error, 3):.3f}]" # Formatting the confidence interval

# Calculates MSE, RMSE, MAE, and R2
def reconstruction_performance(model, dataloader, eval_batches, prefix=""):
    model.eval()
    total_ss_res, total_ss_tot, total_abs_err, total_elements = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for i, (x_batch, _) in enumerate(dataloader):
            if i >= eval_batches: break
            outputs, _ = model(x_batch)
            y_true, y_pred = x_batch.reshape(-1), outputs.reshape(-1)
            total_ss_res += torch.sum((y_true - y_pred) ** 2).item()
            batch_mean = torch.mean(y_true)
            total_ss_tot += torch.sum((y_true - batch_mean) ** 2).item()
            total_abs_err += torch.sum(torch.abs(y_true - y_pred)).item()
            total_elements += y_true.numel()
            
    mse = total_ss_res / total_elements
    rmse = math.sqrt(mse)
    mae = total_abs_err / total_elements
    r2 = 1.0 - (total_ss_res / total_ss_tot) if total_ss_tot != 0 else 0.0
    
    return {f"{prefix}MSE": round(mse, 3), f"{prefix}RMSE": round(rmse, 3), 
            f"{prefix}MAE": round(mae, 3), f"{prefix}R2": round(r2, 3)}

# This function evaluates the latent space using XGBoost and logistic regression for classification
def classification_performance(model, dataset, train_idx, val_idx, global_labels):
    model.eval()
    
    ext_train_loader = DataLoader(dataset, train_idx, batch_size=512, shuffle=False)
    ext_val_loader = DataLoader(dataset, val_idx, batch_size=512, shuffle=False)
    
    # Extracting latents for training and validation
    X_train, X_val = [], []
    with torch.no_grad():
        for xb, _ in ext_train_loader:
            _, latents = model(xb)
            X_train.append(latents.cpu().numpy())
        for xb, _ in ext_val_loader:
            _, latents = model(xb)
            X_val.append(latents.cpu().numpy())
            
    X_train, X_val = np.concatenate(X_train, axis=0), np.concatenate(X_val, axis=0)
    y_train, y_val = global_labels[train_idx], global_labels[val_idx]
    
    num_pos = sum(y_train)
    num_neg = len(y_train) - num_pos
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    # Setting up XGBoost
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic', scale_pos_weight=scale_pos_weight,
        tree_method='hist', n_estimators=150, learning_rate=0.05, max_depth=5,
        eval_metric='auc', random_state=42, n_jobs=-1
    )
    # Fit the model and evaluate AUC and PR-AUC
    xgb_model.fit(X_train, y_train)
    xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_probs)
    xgb_prauc = average_precision_score(y_val, xgb_probs)
    
    # Setting up logistic regression
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1500, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_probs = lr_model.predict_proba(X_val)[:, 1]
    lr_auc = roc_auc_score(y_val, lr_probs)
    lr_prauc = average_precision_score(y_val, lr_probs)
    
    # Saving the metrics in a dictionary
    metrics = {
        "XGB_AUC": round(xgb_auc, 4), "XGB_PRAUC": round(xgb_prauc, 4),
        "LR_AUC": round(lr_auc, 4), "LR_PRAUC": round(lr_prauc, 4)
    }
    return metrics, y_val, xgb_probs, lr_probs

# This function generates reconstruction plots, loss curves, error histograms, best/worst reconstructions, and classification curves, and exports the latent representations
def latent_space_and_plots(model, dataloader, history_dict, plot_dir, run_dir, eval_batches, val_idx, best_clf_data):
  model.eval()
  
  # Random reconstructions
  x_batch, _ = next(iter(dataloader))
  with torch.no_grad():
      reconstructed, _ = model(x_batch)
  
  x_batch_np = x_batch.permute(0, 2, 1).cpu().numpy()
  reconstructed_np = reconstructed.permute(0, 2, 1).cpu().numpy()
  
  fig, axes = plt.subplots(5, 2, figsize=(20, 15))
  axes = axes.flatten()
  for i in range(min(10, len(x_batch_np))):
      axes[i].plot(x_batch_np[i, :, 0], label="Original (Lead I)", alpha=0.7)
      axes[i].plot(reconstructed_np[i, :, 0], label="Reconstruction", color='red', linestyle='--')
      axes[i].set_title(f"Random ECG Sample {i+1}")
      axes[i].legend(loc='upper right')
  plt.tight_layout()
  plt.savefig(os.path.join(plot_dir, "01_10_random_reconstructions.png"))
  plt.close()

  # Loss curve
  loss = history_dict['loss']
  epochs = range(1, len(loss) + 1)
  plt.figure(figsize=(10, 6))
  plt.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
  if 'val_loss' in history_dict:
      plt.plot(epochs, history_dict['val_loss'], 'r--', label='Validation Loss', linewidth=2)
  plt.title('Model Loss Curve (Best Fold)')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.savefig(os.path.join(plot_dir, "02_loss_curve.png"))
  plt.close()

  # Error histogram
  real_ecgs, reconstructed_ecgs, latents = [], [], []
  with torch.no_grad():
      for i, (xb, _) in enumerate(dataloader):
          if i >= eval_batches: break
          out, latent = model(xb)
          real_ecgs.append(xb.permute(0, 2, 1).cpu().numpy())
          reconstructed_ecgs.append(out.permute(0, 2, 1).cpu().numpy())
          latents.append(latent.cpu().numpy())
          
  real_ecgs = np.concatenate(real_ecgs, axis=0)
  reconstructed_ecgs = np.concatenate(reconstructed_ecgs, axis=0)
  latents = np.concatenate(latents, axis=0)
  mse_per_sample = np.mean(np.square(real_ecgs - reconstructed_ecgs), axis=(1, 2))
  
  plt.figure(figsize=(10, 5))
  plt.hist(mse_per_sample, bins=50, color='purple', alpha=0.7, edgecolor='black')
  plt.axvline(np.mean(mse_per_sample), color='red', linestyle='dashed', linewidth=2, label='Mean Error')
  plt.title('Distribution of Reconstruction Errors (MSE)')
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.savefig(os.path.join(plot_dir, "03_error_histogram.png"))
  plt.close()
  
  # Best/Worst Reconstruction
  best_idx = np.argmin(mse_per_sample)  
  worst_idx = np.argmax(mse_per_sample) 
  fig, axes = plt.subplots(1, 2, figsize=(18, 5))
  axes[0].plot(real_ecgs[best_idx, :, 0], label="Original", alpha=0.7)
  axes[0].plot(reconstructed_ecgs[best_idx, :, 0], label="Recon", color='red', linestyle='--')
  axes[0].set_title(f"Best Recon (Error: {mse_per_sample[best_idx]:.4f})")
  axes[1].plot(real_ecgs[worst_idx, :, 0], label="Original", alpha=0.7)
  axes[1].plot(reconstructed_ecgs[worst_idx, :, 0], label="Recon", color='red', linestyle='--')
  axes[1].set_title(f"Worst Recon (Error: {mse_per_sample[worst_idx]:.4f})")
  plt.savefig(os.path.join(plot_dir, "04_best_worst_reconstruction.png"))
  plt.close()

  # Classification curves (ROC and PR)
  if best_clf_data is not None:
      y_val, xgb_probs, lr_probs = best_clf_data

      # ROC Curve Plot
      fpr_xgb, tpr_xgb, _ = roc_curve(y_val, xgb_probs)
      fpr_lr, tpr_lr, _ = roc_curve(y_val, lr_probs)
      
      plt.figure(figsize=(8, 6))
      plt.plot(fpr_xgb, tpr_xgb, color='#1d3557', linewidth=2, label=f'XGBoost (AUC = {roc_auc_score(y_val, xgb_probs):.3f})')
      plt.plot(fpr_lr, tpr_lr, color='#e63946', linewidth=2, label=f'LogReg (AUC = {roc_auc_score(y_val, lr_probs):.3f})')
      plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
      plt.title('ROC Curve (Best Fold Evaluation)')
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.legend(loc='lower right')
      plt.grid(True, linestyle='--', alpha=0.5)
      plt.savefig(os.path.join(plot_dir, "05_roc_curve.png"), dpi=300)
      plt.close()

# PR Curve Plot
      prec_xgb, rec_xgb, _ = precision_recall_curve(y_val, xgb_probs)
      prec_lr, rec_lr, _ = precision_recall_curve(y_val, lr_probs)
      
      # Calculate prevalence baseline
      prevalence = np.mean(y_val)
      
      # Calculate F1 scores across all thresholds
      f1_xgb = np.divide(2 * (prec_xgb * rec_xgb), (prec_xgb + rec_xgb), 
                         out=np.zeros_like(prec_xgb), where=(prec_xgb + rec_xgb) != 0)
      f1_lr = np.divide(2 * (prec_lr * rec_lr), (prec_lr + rec_lr), 
                        out=np.zeros_like(prec_lr), where=(prec_lr + rec_lr) != 0)
      
      # Locate the highest F1 score for XGBoost
      best_idx_xgb = np.argmax(f1_xgb)
      max_f1_xgb = f1_xgb[best_idx_xgb]
      best_rec_xgb = rec_xgb[best_idx_xgb]
      best_prec_xgb = prec_xgb[best_idx_xgb]
      
      # Locate the highest F1 score for LogReg
      best_idx_lr = np.argmax(f1_lr)
      max_f1_lr = f1_lr[best_idx_lr]
      best_rec_lr = rec_lr[best_idx_lr]
      best_prec_lr = prec_lr[best_idx_lr]
      
      plt.figure(figsize=(8, 6))
      
      # Plot the main PR curves
      plt.plot(rec_xgb, prec_xgb, color='#1d3557', linewidth=2, label=f'XGBoost (PR-AUC = {average_precision_score(y_val, xgb_probs):.3f})')
      plt.plot(rec_lr, prec_lr, color='#e63946', linewidth=2, label=f'LogReg (PR-AUC = {average_precision_score(y_val, lr_probs):.3f})')
      
      # Plot the Max F1 points
      plt.plot(best_rec_xgb, best_prec_xgb, marker='o', markersize=9, color='#1d3557', markeredgecolor='white', 
               linestyle='None', label=f'XGB F1 ({max_f1_xgb:.3f})')
      plt.plot(best_rec_lr, best_prec_lr, marker='o', markersize=9, color='#e63946', markeredgecolor='white', 
               linestyle='None', label=f'LogReg F1 ({max_f1_lr:.3f})')
      
      # Add the horizontal prevalence line
      plt.axhline(y=prevalence, color='gray', linestyle=':', linewidth=2, label=f'Prevalence Baseline ({prevalence:.3f})')
      
      plt.title('Precision-Recall Curve (Best Fold Evaluation)')
      plt.xlabel('Recall')
      plt.ylabel('Precision')
      
      # Adjust legend to avoid covering the lines (usually top right or lower left is best for PR curves)
      plt.legend(loc='lower left', fontsize='small')
      plt.grid(True, linestyle='--', alpha=0.5)
      
      plt.savefig(os.path.join(plot_dir, "06_pr_curve.png"), dpi=300)
      plt.close()

  # Export latent space
  processed_val_idx = val_idx[:len(latents)]
  print("      Exporting latent matrices for post-analysis...")
  np.save(os.path.join(run_dir, "saved_latents.npy"), latents)
  np.save(os.path.join(run_dir, "saved_val_idx.npy"), processed_val_idx)

### Main training loop ###
# Loading dataset
full_dataset = MIMIC(h5_file_path=TRAIN_DATA_PATH, seq_len=SEQ_LEN)
TOTAL_AVAILABLE = len(full_dataset.data)

# Extracting AF labels for classification
print("\nExtracting AF labels for classification...")
df_gt_dict = {}
with h5py.File(TRAIN_DATA_PATH, 'r') as f:
    gt_group = f['GT']
    report_cols = [key for key in gt_group.keys() if key.startswith('report_')]
    for col in report_cols:
        df_gt_dict[col] = [val.decode('utf-8') for val in gt_group[col][:]]

# Selecting relevant target
df_gt = pd.DataFrame(df_gt_dict)
EXACT_TARGETS = ["ATRIAL FIBRILLATION", "Atrial fibrillation", "Atrial fibrillation."]
mask = pd.Series(False, index=df_gt.index)
for col in report_cols:
    if col in df_gt.columns:
        mask |= df_gt[col].fillna('').astype(str).str.strip().isin(EXACT_TARGETS)

y_labels_global = mask.astype(int).values
print(f"The labels are loaded. AF-cases: {sum(y_labels_global)}")

print("\n" + "="*60)
print(f"Starting grid search: {K_FOLDS}-Fold CV")
print("="*60)

indices = np.arange(TOTAL_AVAILABLE)
np.random.shuffle(indices)
fold_size = TOTAL_AVAILABLE // K_FOLDS

# Looping through each combination of hyperparameters and save results in separate folders for each run
for idx, p in enumerate(EXPERIMENT_COMBINATIONS):
  readable_date = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
  run_name = f"GridRun_{idx+1:03d}_{datetime.datetime.now().strftime('%d%m_%H%M')}"
  run_dir = os.path.join(OUTPUT_DIR, run_name)
  plot_dir = os.path.join(run_dir, "plots")
  os.makedirs(plot_dir, exist_ok=True)
  
  print("\n" + "="*60)
  print(f"Starting run {idx+1}/{len(EXPERIMENT_COMBINATIONS)}: {run_name}")
  print(f"Testing Parameters: {p}")
  print("="*60)
  
  # Preparing to store metrics
  fold_metrics_list = []
  best_r2 = -float('inf')
  best_fold_history = None
  best_fold_val_loader = None
  best_val_idx = None
  best_clf_data = None
  temp_model_path = os.path.join(run_dir, "temp_best_model.pth")
  
  # Cross-validation loop
  for fold in range(K_FOLDS):
      print(f"\n   >>> Starting fold {fold + 1}/{K_FOLDS}...")
      
      val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
      train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
      
      train_loader = DataLoader(full_dataset, train_idx, p['batch_size'], shuffle=True)
      val_loader = DataLoader(full_dataset, val_idx, p['batch_size'], shuffle=False)
      
      model = Autoencoder(
          SEQ_LEN, IN_CHANNELS, p['latent_dim'], p['base_filters'], p['kernel_size'], 
          p['num_layers'], p['pool_size'], p['activation'], p['dropout_rate'], 
          p['norm_type'], p['pooling_type'], p['masking_ratio']
      ).to(DEVICE) # Initializing the model with the selected hyperparameters and sends it to the GPU if available
      
      optimizer = optim.Adam(model.parameters(), lr=p['learning_rate'])
      criterion = nn.HuberLoss() if p['loss_func'] == 'huber' else nn.MSELoss()
      early_stopper = EarlyStopping(patience=5, delta=0.001)
      history = {'loss': [], 'val_loss': []}
      
      # For each epoch - the model is trained, evaluated, and early stopping is applied
      for epoch in range(150):
          model.train()
          running_loss = 0.0
          for x_batch, y_batch in train_loader:
              optimizer.zero_grad()
              outputs, _ = model(x_batch)
              loss = criterion(outputs, y_batch)
              loss.backward()
              torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
              optimizer.step()
              running_loss += loss.item()
          train_loss = running_loss / len(train_loader)
          
          model.eval()
          val_loss = 0.0
          with torch.no_grad():
              for x_batch, y_batch in val_loader:
                  outputs, _ = model(x_batch)
                  loss = criterion(outputs, y_batch)
                  val_loss += loss.item()
          val_loss /= len(val_loader)
          
          history['loss'].append(train_loss)
          history['val_loss'].append(val_loss)
          if (epoch + 1) % 5 == 0 or epoch == 0:
              print(f"      Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

          early_stopper(val_loss, model)
          if early_stopper.early_stop:
              print(f"      Early stopping triggered at epoch {epoch+1}")
              model.load_state_dict(early_stopper.best_model_state)
              break
      
      eval_batches_metrics = max(1, len(val_idx) // p['batch_size'])
      train_metrics = reconstruction_performance(model, train_loader, eval_batches_metrics, prefix="Train_")
      val_metrics = reconstruction_performance(model, val_loader, eval_batches_metrics, prefix="Val_")
      
      print("      Extracting latents and running XGBoost and LogReg...")
      clf_metrics, y_val_clf, xgb_probs, lr_probs = reconstruction_performance(model, full_dataset, train_idx, val_idx, y_labels_global)
      
      fold_metrics = {**train_metrics, **val_metrics, **clf_metrics}
      fold_metrics["Fold"] = fold + 1
      fold_metrics_list.append(fold_metrics)
      
      print(f"      Fold {fold + 1} R2 -> Train: {train_metrics['Train_R2']:.3f} | Val: {val_metrics['Val_R2']:.3f}")
      print(f"      Fold {fold + 1} XGBoost -> AUC: {clf_metrics['XGB_AUC']:.4f} | PR-AUC: {clf_metrics['XGB_PRAUC']:.4f}")
      print(f"      Fold {fold + 1} LogReg  -> AUC: {clf_metrics['LR_AUC']:.4f} | PR-AUC: {clf_metrics['LR_PRAUC']:.4f}")
      
      if val_metrics['Val_R2'] > best_r2:
          best_r2 = val_metrics['Val_R2']
          torch.save(model.state_dict(), temp_model_path)
          best_fold_history = history
          best_fold_val_loader = val_loader
          best_val_idx = val_idx
          best_clf_data = (y_val_clf, xgb_probs, lr_probs) 
          np.save(os.path.join(run_dir, "saved_train_idx.npy"), train_idx)
          
      del model, optimizer, train_loader, val_loader
      torch.cuda.empty_cache()
      gc.collect()

  avg_metrics = {}
  for metric_name in fold_metrics_list[0].keys():
      if metric_name != "Fold":
          avg_metrics[f"Avg_{metric_name}"] = round(float(np.mean([m[metric_name] for m in fold_metrics_list])), 3)

  # Calculating confidence intervals for the metrics
  ci_val_rmse = confidence_intervals([m["Val_RMSE"] for m in fold_metrics_list])
  ci_xgb_auc = confidence_intervals([m["XGB_AUC"] for m in fold_metrics_list])
  ci_xgb_prauc = confidence_intervals([m["XGB_PRAUC"] for m in fold_metrics_list])
  ci_lr_auc = confidence_intervals([m["LR_AUC"] for m in fold_metrics_list])
  ci_lr_prauc = confidence_intervals([m["LR_PRAUC"] for m in fold_metrics_list])

  print(f"\n   >>> CV finished.")
  print(f"       Avg RMSE:       {avg_metrics.get('Avg_Val_RMSE')} (95% CI: {ci_val_rmse})")
  print(f"       Avg XGB AUC:    {avg_metrics.get('Avg_XGB_AUC')} (95% CI: {ci_xgb_auc})")
  print(f"       Avg LogReg AUC: {avg_metrics.get('Avg_LR_AUC')} (95% CI: {ci_lr_auc})\n")

  best_model = Autoencoder(
      SEQ_LEN, IN_CHANNELS, p['latent_dim'], p['base_filters'], p['kernel_size'],
      p['num_layers'], p['pool_size'], p['activation'], p['dropout_rate'], p['norm_type'], p['pooling_type']
  ).to(DEVICE)
  best_model.load_state_dict(torch.load(temp_model_path, weights_only=True))
  
  plot_eval_batches = max(1, 4000 // p['batch_size'])
  latent_space_and_plots(best_model, best_fold_val_loader, best_fold_history, plot_dir, run_dir, plot_eval_batches, best_val_idx, best_clf_data)
  
  final_model_path = os.path.join(run_dir, "best_fold_model.pth")
  os.rename(temp_model_path, final_model_path)
  
  actual_train_size = TOTAL_AVAILABLE - fold_size
  
  # Saving results to CSV and JSON
  csv_row_dict = {
      "split": f"{actual_train_size:,} / {fold_size:,}",
      "date": readable_date,
      "seq_len": SEQ_LEN,
      "latent_dim": p['latent_dim'],
      "learning_rate": p['learning_rate'],
      "base_filters": p['base_filters'],
      "kernel_size": p['kernel_size'],
      "num_layers": p['num_layers'],
      "pool_size": p['pool_size'],
      "pooling_type": p['pooling_type'],
      "activation": p['activation'],
      "norm_type": p['norm_type'],
      "dropout_rate": p['dropout_rate'],
      "masking_ratio": p['masking_ratio'],
      "batch_size": p['batch_size'],
      "loss_func": p['loss_func'],
      "k_folds": K_FOLDS,
      "Avg_Val_MSE": avg_metrics.get("Avg_Val_MSE"),
      "Avg_Val_RMSE": avg_metrics.get("Avg_Val_RMSE"),
      "CI_Val_RMSE": ci_val_rmse,
      "Avg_Val_MAE": avg_metrics.get("Avg_Val_MAE"),
      "Avg_Val_R2": avg_metrics.get("Avg_Val_R2"),
      "Avg_XGB_AUC": avg_metrics.get("Avg_XGB_AUC"),
      "CI_XGB_AUC": ci_xgb_auc,
      "Avg_XGB_PRAUC": avg_metrics.get("Avg_XGB_PRAUC"),
      "CI_XGB_PRAUC": ci_xgb_prauc,
      "Avg_LR_AUC": avg_metrics.get("Avg_LR_AUC"),
      "CI_LR_AUC": ci_lr_auc,
      "Avg_LR_PRAUC": avg_metrics.get("Avg_LR_PRAUC"),
      "CI_LR_PRAUC": ci_lr_prauc
  }
  
  summary_df = pd.DataFrame([csv_row_dict], columns=list(csv_row_dict.keys()))
  csv_path = os.path.join(OUTPUT_DIR, "experiment_summary.csv")
  summary_df.to_csv(csv_path, mode='a', sep=';', index=False, header=not os.path.exists(csv_path))
  
  with open(os.path.join(run_dir, "config.json"), "w") as f:
      json.dump(csv_row_dict, f, indent=4)
      
  with open(os.path.join(run_dir, "fold_metrics.json"), "w") as f:
      json.dump(fold_metrics_list, f, indent=4)

  print(f"      The run is saved to: {run_dir}\n")

print("\n" + "="*60 + "\nAll experiments are completed\n" + "="*60)