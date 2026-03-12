# ================================
# 1 Imports & Global Config
# ================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import random
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import gc
import h5py
import shutil
import math
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import scipy.stats as st

# Setting seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

# Device configuration & AMP check
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = False

print(f"Using device: {DEVICE}")

# ================================
# 2 Full Grid Search Parameters
# ================================
BASE_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space"
OUTPUT_DIR = os.path.join(BASE_DIR, "Model Development/FullGridSearch")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "Data/Full training dataset/training_dataset.h5")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 5000
IN_CHANNELS = 8
K_FOLDS = 5

GRID_PARAMS = {
  'batch_size': [64],
  'pooling_type': ['average'],
  'latent_dim': [512],
  'learning_rate': [0.0005],
  'base_filters': [32],
  'kernel_size': [25],
  'num_layers': [3], 
  'pool_size': [3], 
  'activation': ['leaky_relu'],
  'norm_type': ['batch'],
  'dropout_rate': [0.0],
  'loss_func': ['huber']
}

keys, values = zip(*GRID_PARAMS.items())
EXPERIMENT_COMBINATIONS = [dict(zip(keys, v)) for v in itertools.product(*values)]

# ================================
# 3 Safe RAM Data Loading & Early Stopping
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
      self.n_batches = len(self.indices) // self.batch_size
      
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
      end = start + self.batch_size
      batch_idx = self.indices[start:end]
      
      x_batch = self.dataset.data[batch_idx].to(DEVICE)
      self.current_batch += 1
      
      return x_batch, x_batch 
      
  def __len__(self):
      return self.n_batches

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

# ================================
# 4 PyTorch Autoencoder Model
# ================================
class ConvAutoencoder(nn.Module):
  def __init__(self, seq_len, in_channels, latent_dim, base_filters, kernel_size,
               num_layers, pool_size, activation, dropout_rate, norm_type, pooling_type):
      super(ConvAutoencoder, self).__init__()
      
      self.in_channels = in_channels
      self.seq_len = seq_len
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
      encoded = self.encoder(x)
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
# 5 Evaluation & Export Functions
# ================================
def evaluate_overall_performance(model, dataloader, eval_batches, prefix=""):
  model.eval()
  y_true, y_pred = [], []
  with torch.no_grad():
      for i, (x_batch, _) in enumerate(dataloader):
          if i >= eval_batches: break
          outputs, _ = model(x_batch)
          y_true.append(x_batch.permute(0, 2, 1).cpu().numpy().flatten())
          y_pred.append(outputs.permute(0, 2, 1).cpu().numpy().flatten())
          
  y_true = np.concatenate(y_true)
  y_pred = np.concatenate(y_pred)
  mse = round(float(mean_squared_error(y_true, y_pred)), 3)
  rmse = round(float(np.sqrt(mse)), 3)
  mae = round(float(mean_absolute_error(y_true, y_pred)), 3)
  r2 = round(float(r2_score(y_true, y_pred)), 3)
  return {f"{prefix}MSE": mse, f"{prefix}RMSE": rmse, f"{prefix}MAE": mae, f"{prefix}R2": r2}

def export_val_data_and_generate_plots(model, dataloader, history_dict, plot_dir, run_dir, eval_batches, val_idx):
  model.eval()
  
  x_batch, _ = next(iter(dataloader))
  with torch.no_grad():
      if USE_AMP:
          with torch.amp.autocast('cuda'):
              reconstructed, _ = model(x_batch)
      else:
          reconstructed, _ = model(x_batch)
  
  x_batch_np = x_batch.permute(0, 2, 1).cpu().numpy()
  reconstructed_np = reconstructed.permute(0, 2, 1).cpu().numpy()
  
  fig, axes = plt.subplots(5, 2, figsize=(20, 15))
  axes = axes.flatten()
  for i in range(min(10, len(x_batch_np))):
      axes[i].plot(x_batch_np[i, :, 0], label="Original (Lead I)", alpha=0.7)
      axes[i].plot(reconstructed_np[i, :, 0], label="Reconstruction", color='red', linestyle='--')
      axes[i].set_title(f"Random ECG Sample {i+1}")
      axes[i].legend()
  plt.tight_layout()
  plt.savefig(os.path.join(plot_dir, "01_10_random_reconstructions.png"))
  plt.close()

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

  processed_val_idx = val_idx[:len(latents)]
  print("      Exporting latent matrices for post-analysis...")
  np.save(os.path.join(run_dir, "saved_latents.npy"), latents)
  np.save(os.path.join(run_dir, "saved_val_idx.npy"), processed_val_idx)

# ================================
# 6 THE EXPERIMENT EXECUTION
# ================================
full_dataset = ECGDataset(h5_file_path=TRAIN_DATA_PATH)
TOTAL_AVAILABLE = len(full_dataset.data)

print("\n" + "="*60)
print(f"STARTING FULL GRID EXPERIMENT: STANDARD {K_FOLDS}-FOLD CV")
print("="*60)

# We define the indices and shuffle them ONCE for the entire standard K-Fold
indices = np.arange(TOTAL_AVAILABLE)
np.random.shuffle(indices)
fold_size = TOTAL_AVAILABLE // K_FOLDS

for idx, p in enumerate(EXPERIMENT_COMBINATIONS):
  readable_date = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
  run_name = f"GridRun_{idx+1:03d}_{datetime.datetime.now().strftime('%d%m_%H%M')}"
  run_dir = os.path.join(OUTPUT_DIR, run_name)
  plot_dir = os.path.join(run_dir, "plots")
  os.makedirs(plot_dir, exist_ok=True)
  
  print("\n" + "="*60)
  print(f"STARTING RUN {idx+1}/{len(EXPERIMENT_COMBINATIONS)}: {run_name}")
  print(f"Testing Parameters: {p}")
  print("="*60)
  
  fold_metrics_list = []
  best_r2 = -float('inf')
  best_fold_history = None
  best_fold_val_loader = None
  best_val_idx = None
  temp_model_path = os.path.join(run_dir, "temp_best_model.pth")
  
  for fold in range(K_FOLDS):
      print(f"\n   >>> Starting Fold {fold + 1}/{K_FOLDS} (Standard Split)...")
      
      # Strict, non-overlapping sequential chunks for K-Fold
      val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
      train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
      
      train_loader = FastTensorDataLoader(full_dataset, train_idx, p['batch_size'], shuffle=True)
      val_loader = FastTensorDataLoader(full_dataset, val_idx, p['batch_size'], shuffle=False)
      
      model = ConvAutoencoder(
          SEQ_LEN, IN_CHANNELS, p['latent_dim'], p['base_filters'], p['kernel_size'], 
          p['num_layers'], p['pool_size'], p['activation'], p['dropout_rate'], p['norm_type'], p['pooling_type']
      ).to(DEVICE)
      
      optimizer = optim.Adam(model.parameters(), lr=p['learning_rate'])
      criterion = nn.HuberLoss() if p['loss_func'] == 'huber' else nn.MSELoss()
      early_stopper = EarlyStopping(patience=5, delta=0.001)
      history = {'loss': [], 'val_loss': []}
      
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
      train_metrics = evaluate_overall_performance(model, train_loader, eval_batches_metrics, prefix="Train_")
      val_metrics = evaluate_overall_performance(model, val_loader, eval_batches_metrics, prefix="Val_")
      fold_metrics = {**train_metrics, **val_metrics}
      fold_metrics["Fold"] = fold + 1
      fold_metrics_list.append(fold_metrics)
      
      print(f"      Fold {fold + 1} R2 -> Train: {train_metrics['Train_R2']:.3f} | Val: {val_metrics['Val_R2']:.3f}")
      
      if val_metrics['Val_R2'] > best_r2:
          best_r2 = val_metrics['Val_R2']
          torch.save(model.state_dict(), temp_model_path)
          best_fold_history = history
          best_fold_val_loader = val_loader
          best_val_idx = val_idx
          
      del model, optimizer, train_loader, val_loader
      torch.cuda.empty_cache()
      gc.collect()

  avg_metrics = {}
  val_rmse_list = [m["Val_RMSE"] for m in fold_metrics_list]
  for metric_name in fold_metrics_list[0].keys():
      if metric_name != "Fold":
          avg_metrics[f"Avg_{metric_name}"] = round(float(np.mean([m[metric_name] for m in fold_metrics_list])), 3)
          avg_metrics[f"Std_{metric_name}"] = round(float(np.std([m[metric_name] for m in fold_metrics_list], ddof=1)), 3)
  
  n_folds = len(val_rmse_list)
  mean_rmse = np.mean(val_rmse_list)
  std_err = np.std(val_rmse_list, ddof=1) / np.sqrt(n_folds)
  margin_of_error = st.t.ppf(1 - 0.025, n_folds - 1) * std_err if n_folds > 1 else 0
  ci_val_rmse = f"[{round(mean_rmse - margin_of_error, 3):.3f}, {round(mean_rmse + margin_of_error, 3):.3f}]"

  print(f"\n   >>> CV Finished. Average Val RMSE: {avg_metrics.get('Avg_Val_RMSE')} (95% CI: {ci_val_rmse})\n")

  best_model = ConvAutoencoder(
      SEQ_LEN, IN_CHANNELS, p['latent_dim'], p['base_filters'], p['kernel_size'],
      p['num_layers'], p['pool_size'], p['activation'], p['dropout_rate'], p['norm_type'], p['pooling_type']
  ).to(DEVICE)
  best_model.load_state_dict(torch.load(temp_model_path, weights_only=True))
  
  plot_eval_batches = max(1, 4000 // p['batch_size'])
  export_val_data_and_generate_plots(best_model, best_fold_val_loader, best_fold_history, plot_dir, run_dir, plot_eval_batches, best_val_idx)
  
  final_model_path = os.path.join(run_dir, "best_fold_model.pth")
  os.rename(temp_model_path, final_model_path)

  # --- EXACT CSV COLUMN MAPPING ---
  actual_train_size = TOTAL_AVAILABLE - fold_size
  
  csv_row_dict = {
      "split": f"{actual_train_size:,} / {fold_size:,}",
      "date": readable_date,
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
      "batch_size": p['batch_size'],
      "loss_func": p['loss_func'],
      "k_folds": K_FOLDS,
      "Pruned": False,
      "Avg_Val_MSE": avg_metrics.get("Avg_Val_MSE"),
      "Avg_Val_RMSE": avg_metrics.get("Avg_Val_RMSE"),
      "Std_Val_RMSE": avg_metrics.get("Std_Val_RMSE"),
      "CI_Val_RMSE": ci_val_rmse,
      "Avg_Val_MAE": avg_metrics.get("Avg_Val_MAE"),
      "Avg_Val_R2": avg_metrics.get("Avg_Val_R2")
  }
  
  summary_df = pd.DataFrame([csv_row_dict], columns=list(csv_row_dict.keys()))
  csv_path = os.path.join(OUTPUT_DIR, "experiment_summary.csv")
  summary_df.to_csv(csv_path, mode='a', sep=';', index=False, header=not os.path.exists(csv_path))
  
  with open(os.path.join(run_dir, "config.json"), "w") as f:
      json.dump(csv_row_dict, f, indent=4)
      
  with open(os.path.join(run_dir, "fold_metrics.json"), "w") as f:
      json.dump(fold_metrics_list, f, indent=4)

  print(f"      EXPERIMENT SAVED TO CSV AND DIRECTORY: {run_dir}\n")

print("\n" + "="*60 + "\nALL EXPERIMENTS COMPLETED!\n" + "="*60)