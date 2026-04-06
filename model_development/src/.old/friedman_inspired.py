# ================================
# 1 Imports & Global Config
# ================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import random
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import gc
import h5py
import math
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import torch.nn.functional as F

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
print(f"Using device: {DEVICE}")

# ================================
# 2 Full Grid Search Parameters
# ================================
BASE_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space"
OUTPUT_DIR = os.path.join(BASE_DIR, "model_development/full_grid_search")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data/full_training_set/training_dataset.h5")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 5000
IN_CHANNELS = 8
K_FOLDS = 2

GRID_PARAMS = {
    'batch_size': [512],
    'pooling_type': ['max'],
    'latent_dim': [512],
    'learning_rate': [0.0005],
    'base_filters': [64],
    'kernel_size': [35],
    'num_layers': [4], 
    'pool_size': [4, 5], 
    'activation': ['leaky_relu'],
    'norm_type': ['batch'],
    'dropout_rate': [0.0],
    'noise_factor': [0.3]
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
        
        print("Extracting Clinical Labels for Latent Probing...")
        df_gt = pd.read_hdf(h5_file_path, key='GT')
        report_cols = [f'report_{i}' for i in range(18)]
        
        target_list = ["ATRIAL FIBRILLATION", "Atrial fibrillation", "Atrial fibrillation."]
        mask = pd.Series(False, index=df_gt.index)
        
        for col in report_cols:
            if col in df_gt.columns:
                col_cleaned = df_gt[col].fillna('').astype(str).str.strip()
                mask |= col_cleaned.isin(target_list)
        
        self.labels = torch.tensor(mask.astype(int).values, dtype=torch.long)
        print(f"Found {self.labels.sum().item()} AFib cases for Latent Evaluation.")
        
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
        labels_batch = self.dataset.labels[batch_idx]
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
# 4 PyTorch SimCLR Model & Loss
# ================================
class ContrastiveEncoder(nn.Module):
    def __init__(self, seq_len, in_channels, latent_dim, base_filters, kernel_size, num_layers, pool_size, norm_type, activation, dropout_rate, pooling_type):
        super(ContrastiveEncoder, self).__init__()
        
        encoder_layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            filters = base_filters * (2**i)
            conv_stride = pool_size if pooling_type == 'stride' else 1
            
            current_kernel = max(kernel_size // (2**i), 5)
            if current_kernel % 2 == 0: 
                current_kernel += 1
            padding = current_kernel // 2 
            
            encoder_layers.append(nn.Conv1d(current_channels, filters, current_kernel, stride=conv_stride, padding=padding))
            
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
        self.flattened_size = int(np.prod(self.encoder(dummy_input).shape[1:]))
        
        self.representation = nn.Linear(self.flattened_size, latent_dim)
        
        self.projection_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim // 2)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        flattened = encoded.view(encoded.size(0), -1)
        h = self.representation(flattened)
        z = self.projection_head(h)
        return h, z

def nt_xent_loss(z_i, z_j, temperature=0.1):
    """SimCLR Contrastive Loss Function"""
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)
    
    sim_matrix = torch.matmul(z, z.T) / temperature
    
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)
    
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim_matrix.masked_fill_(mask, -9e15)
    
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

# ================================
# 5 Evaluation Functions
# ================================
def evaluate_latent_space(model, train_loader, val_loader, eval_batches=50):
    model.eval()
    
    def get_latents(dataloader):
        latents, labels = [], []
        with torch.no_grad():
            for i, (x_batch, y_labels) in enumerate(dataloader):
                if i >= eval_batches: break
                h, _ = model(x_batch)
                latents.append(h.cpu().numpy())
                labels.append(y_labels.numpy())
        return np.concatenate(latents), np.concatenate(labels)

    X_train, y_train = get_latents(train_loader)
    X_val, y_val = get_latents(val_loader)
    
    if len(np.unique(y_val)) < 2 or len(np.unique(y_train)) < 2:
        return 0.5 
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=10, random_state=42)
    clf.fit(X_train_scaled, y_train)
    probs = clf.predict_proba(X_val_scaled)[:, 1]
    
    auc = roc_auc_score(y_val, probs)
    return round(float(auc), 3)

def export_basic_plots(history_dict, plot_dir, run_dir, model, val_loader, eval_batches, val_idx):
    """Stripped down to only plot loss and save latents, since there are no reconstructions"""
    loss = history_dict['loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'b-', label='Train Contrastive Loss', linewidth=2)
    if 'val_loss' in history_dict:
        plt.plot(epochs, history_dict['val_loss'], 'r--', label='Val Contrastive Loss', linewidth=2)
    plt.title('SimCLR Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, "01_loss_curve.png"))
    plt.close()

    model.eval()
    latents = []
    with torch.no_grad():
        for i, (xb, _) in enumerate(val_loader):
            if i >= eval_batches: break
            h, _ = model(xb)
            latents.append(h.cpu().numpy())
            
    latents = np.concatenate(latents, axis=0)
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
print(f"STARTING FULL GRID EXPERIMENT: STANDARD {K_FOLDS}-FOLD CV (SimCLR)")
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
    best_latent_auc = -float('inf') 
    best_fold_history = None
    best_fold_val_loader = None
    best_val_idx = None
    temp_model_path = os.path.join(run_dir, "temp_best_model.pth")
    
    for fold in range(K_FOLDS):
        print(f"\n   >>> Starting Fold {fold + 1}/{K_FOLDS}...")
        
        val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
        
        train_loader = FastTensorDataLoader(full_dataset, train_idx, p['batch_size'], shuffle=True)
        val_loader = FastTensorDataLoader(full_dataset, val_idx, p['batch_size'], shuffle=False)
        
        model = ContrastiveEncoder(
            SEQ_LEN, IN_CHANNELS, p['latent_dim'], p['base_filters'], p['kernel_size'], 
            p['num_layers'], p['pool_size'], p['norm_type'], p['activation'], p['dropout_rate'], p['pooling_type']
        ).to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=p['learning_rate'])
        
        early_stopper = EarlyStopping(patience=15, delta=0.001)
        history = {'loss': [], 'val_loss': [], 'latent_auc': []}
        eval_batches_metrics = min(100, max(1, len(val_idx) // p['batch_size']))
        
        for epoch in range(150):
            model.train()
            running_loss = 0.0
            
            for x_batch, _ in train_loader:
                optimizer.zero_grad()
                
                # View 1: Gaussian Noise
                x_view1 = x_batch + torch.randn_like(x_batch) * p['noise_factor']
                
                # View 2: Contiguous Masking
                x_view2 = x_batch.clone()
                drop_length = int(SEQ_LEN * p['noise_factor'])
                start_indices = torch.randint(0, SEQ_LEN - drop_length, (x_batch.size(0),))
                for j in range(x_batch.size(0)):
                    x_view2[j, :, start_indices[j]:start_indices[j] + drop_length] = 0.0
                
                _, z1 = model(x_view1)
                _, z2 = model(x_view2)
                
                loss = nt_xent_loss(z1, z2)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()
            train_loss = running_loss / len(train_loader)
            
            # --- EVALUATION ---
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, _ in val_loader:
                    # Calculate Val Loss using the same corruptions
                    x_view1 = x_batch + torch.randn_like(x_batch) * p['noise_factor']
                    x_view2 = x_batch.clone()
                    start_indices = torch.randint(0, SEQ_LEN - drop_length, (x_batch.size(0),))
                    for j in range(x_batch.size(0)):
                        x_view2[j, :, start_indices[j]:start_indices[j] + drop_length] = 0.0
                        
                    _, z1 = model(x_view1)
                    _, z2 = model(x_view2)
                    loss = nt_xent_loss(z1, z2)
                    val_loss += loss.item()
            val_loss /= len(val_loader)
            
            current_auc = evaluate_latent_space(model, train_loader, val_loader, eval_batches=eval_batches_metrics)
            
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['latent_auc'].append(current_auc)
            
            print(f"      Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Latent AUC: {current_auc:.3f}")

            early_stopper(current_auc, model)
            if early_stopper.early_stop:
                print(f"      Early stopping triggered at epoch {epoch+1}")
                model.load_state_dict(early_stopper.best_model_state)
                break
        
        # Final Fold Metrics
        final_auc = evaluate_latent_space(model, train_loader, val_loader, eval_batches=eval_batches_metrics)
        
        fold_metrics = {"Val_Latent_AUC": final_auc, "Fold": fold + 1}
        fold_metrics_list.append(fold_metrics)
        
        print(f"      Fold {fold + 1} -> Latent AUC: {final_auc:.3f}")
        
        if final_auc > best_latent_auc:
            best_latent_auc = final_auc
            torch.save(model.state_dict(), temp_model_path)
            best_fold_history = history
            best_fold_val_loader = val_loader
            best_val_idx = val_idx
            
        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    avg_metrics = {}
    for metric_name in fold_metrics_list[0].keys():
        if metric_name != "Fold":
            avg_metrics[f"Avg_{metric_name}"] = round(float(np.mean([m[metric_name] for m in fold_metrics_list])), 3)
            
    print(f"\n   >>> CV Finished. Average Latent AUC: {avg_metrics.get('Avg_Val_Latent_AUC'):.3f}\n")

    # Export basic plots and latents for the best fold
    best_model = ContrastiveEncoder(
        SEQ_LEN, IN_CHANNELS, p['latent_dim'], p['base_filters'], p['kernel_size'], 
        p['num_layers'], p['pool_size'], p['norm_type'], p['activation'], p['dropout_rate'], p['pooling_type']
    ).to(DEVICE)
    best_model.load_state_dict(torch.load(temp_model_path, weights_only=True))
    export_basic_plots(best_fold_history, plot_dir, run_dir, best_model, best_fold_val_loader, eval_batches_metrics, best_val_idx)

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
        "dropout_rate": p['dropout_rate'],
        "batch_size": p['batch_size'],
        "noise_factor": p['noise_factor'],
        "k_folds": K_FOLDS,
        "Avg_Val_Latent_AUC": avg_metrics.get("Avg_Val_Latent_AUC")
    }
    
    summary_df = pd.DataFrame([csv_row_dict], columns=list(csv_row_dict.keys()))
    csv_path = os.path.join(OUTPUT_DIR, "experiment_summary.csv")
    summary_df.to_csv(csv_path, mode='a', sep=';', index=False, header=not os.path.exists(csv_path))

    print(f"      EXPERIMENT SAVED TO CSV AND DIRECTORY: {run_dir}\n")

print("\n" + "="*60 + "\nALL EXPERIMENTS COMPLETED!\n" + "="*60)