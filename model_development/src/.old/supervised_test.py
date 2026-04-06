# ================================
# 1 Imports & Global Config
# ================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import gc
import h5py
import math
import torch
import torch.nn as nn
import torch.optim as optim
import itertools

# Setting seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ================================
# 2 Full Grid Search Parameters
# ================================
BASE_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space"
OUTPUT_DIR = os.path.join(BASE_DIR, "model_development/supervised_ceiling_test")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data/full_training_set/training_dataset.h5")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 5000
IN_CHANNELS = 8
K_FOLDS = 2

GRID_PARAMS = {
    'batch_size': [256], 
    'pooling_type': ['max'],
    'latent_dim': [512],
    'learning_rate': [0.0005],
    'base_filters': [64],
    'kernel_size': [35],
    'num_layers': [3],  
    'pool_size': [2], 
    'activation': ['leaky_relu'],
    'norm_type': ['batch'],
    'dropout_rate': [0.2] 
}

keys, values = zip(*GRID_PARAMS.items())
EXPERIMENT_COMBINATIONS = [dict(zip(keys, v)) for v in itertools.product(*values)]

# ================================
# 3 Safe RAM Data Loading & Labels
# ================================
class ECGDataset:
    def __init__(self, h5_file_path):
        print(f"\nLoading entire dataset directly into SYSTEM RAM from {h5_file_path}...")
        with h5py.File(h5_file_path, 'r') as h5f:
            self.data = torch.tensor(h5f['rhythm_filtered'][:], dtype=torch.float32).permute(0, 2, 1)
        print(f"Dataset loaded to CPU RAM. Shape: {self.data.shape}")
        
        print("Extracting Clinical Labels for Supervised Training...")
        df_gt = pd.read_hdf(h5_file_path, key='GT')
        
        print("Aligning labels to HDF5 array index...")
        df_gt = df_gt.sort_values(by='h5idx').reset_index(drop=True)
        
        report_cols = [f'report_{i}' for i in range(18)]
        
        target_list = ["ATRIAL FIBRILLATION", "Atrial fibrillation", "Atrial fibrillation."]
        mask = pd.Series(False, index=df_gt.index)
        
        for col in report_cols:
            if col in df_gt.columns:
                col_cleaned = df_gt[col].fillna('').astype(str).str.strip()
                mask |= col_cleaned.isin(target_list)
        
        self.labels = torch.tensor(mask.astype(int).values, dtype=torch.long)
        print(f"Found {self.labels.sum().item()} AFib cases.")
        
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
        labels_batch = self.dataset.labels[batch_idx].to(DEVICE)
        self.current_batch += 1
        
        return x_batch, labels_batch 
        
    def __len__(self):
        return self.n_batches

class EarlyStopping:
    def __init__(self, patience=15, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, current_score, model):
        if math.isnan(current_score):
            self.early_stop = True
            return
        if self.best_score is None:
            self.best_score = current_score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif current_score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

# ================================
# 4 PyTorch Supervised Model (Global Pooling)
# ================================
class CRNN_Encoder(nn.Module):
    def __init__(self, seq_len, in_channels, base_filters, kernel_size, num_layers, pool_size, norm_type, activation, dropout_rate, pooling_type):
        super(CRNN_Encoder, self).__init__()
        
        encoder_layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            filters = base_filters * (2**i)
            conv_stride = pool_size if pooling_type == 'stride' else 1
            
            current_kernel = max(kernel_size // (2**i), 5)
            if current_kernel % 2 == 0: current_kernel += 1
            padding = current_kernel // 2 
            
            encoder_layers.append(nn.Conv1d(current_channels, filters, current_kernel, stride=conv_stride, padding=padding))
            
            if norm_type == 'layer':
                encoder_layers.append(nn.GroupNorm(1, filters))
            elif norm_type == 'batch':
                encoder_layers.append(nn.BatchNorm1d(filters))
                
            encoder_layers.append(nn.LeakyReLU() if activation == 'leaky_relu' else nn.ReLU())
            
            if pooling_type == 'max':
                encoder_layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
            elif pooling_type == 'average':
                encoder_layers.append(nn.AvgPool1d(kernel_size=pool_size, stride=pool_size))
                
            if dropout_rate > 0.0:
                encoder_layers.append(nn.Dropout(dropout_rate))
                
            current_channels = filters
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(current_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1) 
        )

    def forward(self, x):
        # Shape: [Batch, Channels, Time]
        cnn_features = self.encoder(x) 
        
        # Shape: [Batch, Channels, 1]
        pooled_features = self.global_pool(cnn_features) 
        
        # Shape: [Batch, Channels]
        flat_features = pooled_features.view(pooled_features.size(0), -1)
        
        # Shape: [Batch]
        logits = self.classifier(flat_features)
        return logits.squeeze()

# ================================
# 5 Simple Plotting Export
# ================================
def export_loss_curve(history_dict, plot_dir):
    loss = history_dict['loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'b-', label='Train BCE Loss', linewidth=2)
    if 'val_loss' in history_dict:
        plt.plot(epochs, history_dict['val_loss'], 'r--', label='Val BCE Loss', linewidth=2)
    plt.title('Supervised Classification Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, "01_loss_curve.png"))
    plt.close()

# ================================
# 6 THE EXPERIMENT EXECUTION
# ================================
full_dataset = ECGDataset(h5_file_path=TRAIN_DATA_PATH)
TOTAL_AVAILABLE = len(full_dataset.data)

print("\n" + "="*60)
print(f"STARTING FULL GRID EXPERIMENT: SUPERVISED CEILING TEST")
print("="*60)

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
    best_val_auc = -float('inf') 
    best_fold_history = None
    temp_model_path = os.path.join(run_dir, "temp_best_model.pth")
    
    for fold in range(K_FOLDS):
        print(f"\n   >>> Starting Fold {fold + 1}/{K_FOLDS}...")
        
        val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
        
        train_loader = FastTensorDataLoader(full_dataset, train_idx, p['batch_size'], shuffle=True)
        val_loader = FastTensorDataLoader(full_dataset, val_idx, p['batch_size'], shuffle=False)
        
        model = CRNN_Encoder(
            SEQ_LEN, IN_CHANNELS, p['base_filters'], p['kernel_size'], 
            p['num_layers'], p['pool_size'], p['norm_type'], p['activation'], p['dropout_rate'], p['pooling_type']
        ).to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=p['learning_rate'])
        num_positives = full_dataset.labels.sum().item()
        num_negatives = len(full_dataset.labels) - num_positives
        pos_weight = torch.tensor([num_negatives / num_positives], dtype=torch.float32).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        early_stopper = EarlyStopping(patience=10, delta=0.001)
        history = {'loss': [], 'val_loss': [], 'val_auc': []}
        
        for epoch in range(150):
            model.train()
            running_loss = 0.0
            
            for x_batch, labels_batch in train_loader:
                optimizer.zero_grad()
                probs = model(x_batch)
                loss = criterion(probs, labels_batch.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()
                
            train_loss = running_loss / len(train_loader)
            
            # --- EVALUATION ---
            model.eval()
            val_loss = 0.0
            all_probs = []
            all_labels = []
            
            with torch.no_grad():
                for x_batch, labels_batch in val_loader:
                    probs = model(x_batch)
                    loss = criterion(probs, labels_batch.float())
                    val_loss += loss.item()
                    
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(labels_batch.cpu().numpy())
                    
            val_loss /= len(val_loader)
            
            current_auc = roc_auc_score(all_labels, all_probs)
            
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_auc'].append(current_auc)
            
            print(f"      Epoch {epoch+1:03d} | Train BCE: {train_loss:.4f} | Val BCE: {val_loss:.4f} | Supervised AUC: {current_auc:.3f}")

            early_stopper(current_auc, model)
            if early_stopper.early_stop:
                print(f"      Early stopping triggered at epoch {epoch+1}")
                model.load_state_dict(early_stopper.best_model_state)
                break
        
        fold_metrics = {"Val_AUC": early_stopper.best_score, "Fold": fold + 1}
        fold_metrics_list.append(fold_metrics)
        
        print(f"      Fold {fold + 1} -> Best Supervised AUC: {early_stopper.best_score:.3f}")
        
        if early_stopper.best_score > best_val_auc:
            best_val_auc = early_stopper.best_score
            torch.save(model.state_dict(), temp_model_path)
            best_fold_history = history
            
        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    avg_metrics = {}
    for metric_name in fold_metrics_list[0].keys():
        if metric_name != "Fold":
            avg_metrics[f"Avg_{metric_name}"] = round(float(np.mean([m[metric_name] for m in fold_metrics_list])), 3)
            
    print(f"\n   >>> CV Finished. Average Supervised AUC: {avg_metrics.get('Avg_Val_AUC'):.3f}\n")

    # Export basic plots for the best fold
    export_loss_curve(best_fold_history, plot_dir)

    # Save logic and CSV export
    csv_row_dict = {
        "split": f"{TOTAL_AVAILABLE - fold_size:,} / {fold_size:,}",
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
        "batch_size": p['batch_size'],
        "k_folds": K_FOLDS,
        "Avg_Val_Supervised_AUC": avg_metrics.get("Avg_Val_AUC")
    }
    
    summary_df = pd.DataFrame([csv_row_dict], columns=list(csv_row_dict.keys()))
    csv_path = os.path.join(OUTPUT_DIR, "supervised_ceiling_summary.csv")
    summary_df.to_csv(csv_path, mode='a', sep=';', index=False, header=not os.path.exists(csv_path))

    print(f"      EXPERIMENT SAVED TO CSV AND DIRECTORY: {run_dir}\n")

print("\n" + "="*60 + "\nALL EXPERIMENTS COMPLETED!\n" + "="*60)