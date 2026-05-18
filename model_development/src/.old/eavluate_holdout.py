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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

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
    """Passes data through the model to extract latents and calc reconstruction metrics."""
    model.eval()
    latents_list = []
    total_ss_res, total_ss_tot, total_abs_err, total_elements = 0.0, 0.0, 0.0, 0
    
    with torch.no_grad():
        for x_batch in dataloader:
            outputs, latents = model(x_batch)
            latents_list.append(latents.cpu().numpy())
            
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
    
    metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
    return np.concatenate(latents_list, axis=0), metrics

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
    
    X_holdout, holdout_recon_metrics = get_latents_and_reconstruction(model, holdout_loader)

    # 6. Evaluate Classification on Holdout
    xgb_probs = xgb_model.predict_proba(X_holdout)[:, 1]
    lr_probs = lr_model.predict_proba(X_holdout)[:, 1]

    holdout_clf_metrics = {
        "XGB_AUC": roc_auc_score(y_holdout, xgb_probs),
        "XGB_PRAUC": average_precision_score(y_holdout, xgb_probs),
        "LR_AUC": roc_auc_score(y_holdout, lr_probs),
        "LR_PRAUC": average_precision_score(y_holdout, lr_probs)
    }

    # 7. Print Final Results
    print("\n" + "="*60)
    print(" HOLDOUT EVALUATION RESULTS ")
    print("="*60)
    print("Reconstruction Metrics:")
    print(f"  RMSE: {holdout_recon_metrics['RMSE']:.4f}")
    print(f"  R2:   {holdout_recon_metrics['R2']:.4f}")
    print(f"  MSE:  {holdout_recon_metrics['MSE']:.4f}")
    print("\nClassification Metrics:")
    print(f"  XGBoost AUC:    {holdout_clf_metrics['XGB_AUC']:.4f}")
    print(f"  XGBoost PR-AUC: {holdout_clf_metrics['XGB_PRAUC']:.4f}")
    print(f"  LogReg AUC:     {holdout_clf_metrics['LR_AUC']:.4f}")
    print(f"  LogReg PR-AUC:  {holdout_clf_metrics['LR_PRAUC']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()