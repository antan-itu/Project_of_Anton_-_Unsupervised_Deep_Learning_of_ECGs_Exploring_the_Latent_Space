# ================================
# 1 Imports & Global Config
# ================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import h5py
import math
import torch
import torch.nn as nn
import torch.optim as optim

# Setting seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ================================
# 2 The "Champion" Hyperparameters
# ================================
TOTAL_SAMPLES = 10000 
SEQ_LEN = 5000
IN_CHANNELS = 8

# Locked Champion Parameters from Optuna Trial
LATENT_DIM = 512
LR = 0.0005
BASE_FILTERS = 32
KERNEL_SIZE = 9
NUM_LAYERS = 3
STRIDE_SIZE = 3
POOLING_TYPE = 'max'
ACTIVATION = 'leaky_relu'
NORM_TYPE = 'batch'
DROPOUT_RATE = 0.0
BATCH_SIZE = 128
LOSS_FUNC = 'huber'

# The sizes of training data we want to test
TRAIN_SIZES = [9000, 6000, 3000, 2000, 1000, 750, 500, 350, 200, 100, 50] 
VAL_SIZE = 1000

# Paths (Network Drive)
BASE_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space"
OUTPUT_DIR = os.path.join(BASE_DIR, "Model Development/HowMuchDataIsEnough")
H5_DATA_PATH = os.path.join(BASE_DIR, "Data/MIMIC-IV_Subset/ecg_dataset_10k_8lead.h5")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# 3 Data Loading & Utilities
# ================================
class ECGDataset:
    def __init__(self, h5_file_path):
        print(f"Loading HDF5 dataset directly into VRAM from {h5_file_path}...")
        with h5py.File(h5_file_path, 'r') as h5f:
            self.data = torch.tensor(h5f['ecg_data'][:], dtype=torch.float32).to(DEVICE)
        print(f"Dataset loaded. Shape: {self.data.shape}")
        
        means = self.data.mean(dim=2, keepdim=True)
        stds = self.data.std(dim=2, keepdim=True)
        self.data = (self.data - means) / (stds + 1e-8)
        print("Data standardized.")

class FastTensorDataLoader:
    def __init__(self, dataset, indices, batch_size, shuffle=False):
        self.dataset = dataset
        self.indices = torch.tensor(indices, dtype=torch.long, device=DEVICE)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = len(self.indices) // self.batch_size 
        
    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(len(self.indices), device=DEVICE)
            self.indices = self.indices[perm]
        self.current_batch = 0
        return self
        
    def __next__(self):
        if self.current_batch >= self.n_batches:
            raise StopIteration
        start = self.current_batch * self.batch_size
        end = start + self.batch_size
        batch_idx = self.indices[start:end]
        x_batch = self.dataset.data[batch_idx]
        self.current_batch += 1
        return x_batch, x_batch 

    def __len__(self):
        return self.n_batches

class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
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

def evaluate_overall_performance(model, dataloader, eval_batches):
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
    
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, r2

# ================================
# 4 PyTorch Autoencoder Model
# ================================
class ConvAutoencoder(nn.Module):
    def __init__(self, seq_len, in_channels, latent_dim, base_filters, kernel_size, 
                 num_layers, stride_size, activation, dropout_rate, norm_type, pooling_type):
        super(ConvAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        padding = kernel_size // 2 
        
        encoder_layers = []
        current_channels = in_channels
        for i in range(num_layers):
            filters = base_filters * (2**i)
            conv_stride = stride_size if pooling_type == 'stride' else 1
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
                encoder_layers.append(nn.MaxPool1d(kernel_size=stride_size, stride=stride_size))
            elif pooling_type == 'average':
                encoder_layers.append(nn.AvgPool1d(kernel_size=stride_size, stride=stride_size))
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
                decoder_layers.append(nn.Upsample(scale_factor=stride_size))
                conv_stride = 1
            else:
                conv_stride = stride_size
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
# 5 The Experiment Execution
# ================================
print("\n" + "="*50)
print("STARTING LEARNING CURVE EXPERIMENT")
print("="*50)

full_dataset = ECGDataset(h5_file_path=H5_DATA_PATH)

# Shuffle all 10,000 indices securely
all_indices = np.random.permutation(TOTAL_SAMPLES)

# 1. Isolate the Global Validation Set (Strictly 2,000 samples)
val_indices = all_indices[:VAL_SIZE]
val_loader = FastTensorDataLoader(full_dataset, val_indices, BATCH_SIZE, shuffle=False)

# 2. Define the remaining 8,000 as the Training Pool
train_pool = all_indices[VAL_SIZE:]

results_list = []

for size in TRAIN_SIZES:
    print(f"\n>>> Training with {size} ECGs... (Validating on fixed {VAL_SIZE} ECGs)")
    
    current_batch_size = min(BATCH_SIZE, size)
    current_train_idx = train_pool[:size]
    train_loader = FastTensorDataLoader(full_dataset, current_train_idx, current_batch_size, shuffle=True)
    
    model = ConvAutoencoder(
        SEQ_LEN, IN_CHANNELS, LATENT_DIM, BASE_FILTERS, KERNEL_SIZE, 
        NUM_LAYERS, STRIDE_SIZE, ACTIVATION, DROPOUT_RATE, NORM_TYPE, POOLING_TYPE
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.HuberLoss() if LOSS_FUNC == 'huber' else nn.MSELoss()
    early_stopper = EarlyStopping(patience=10, delta=0.001) 
    
    # Dictionary to track the loss over epochs
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(500):
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
            
        # Eval
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs, _ = model(x_batch)
                batch_loss = criterion(outputs, y_batch)
                val_loss += batch_loss.item()
        val_loss /= len(val_loader)
        
        # Save the losses to history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"      Epoch {epoch+1:03d}/500 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"      Early stopping triggered at epoch {epoch+1}. Best Val Loss: {early_stopper.best_loss:.4f}")
            model.load_state_dict(early_stopper.best_model_state)
            break
            
    # Calculate final scores over the entire validation loader
    val_rmse, val_r2 = evaluate_overall_performance(model, val_loader, eval_batches=len(val_loader))
    print(f"    Final Val RMSE: {val_rmse:.3f} | Final Val R2: {val_r2:.3f}")
    
    results_list.append({
        "Train_Size": size,
        "Val_RMSE": val_rmse,
        "Val_R2": val_r2
    })
    
    # Plot the Loss Curve for this size
    plt.figure(figsize=(8, 5))
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs_range, history['val_loss'], label='Val Loss', color='red', linestyle='--', linewidth=2)
    plt.title(f'Loss Curve - Trained on {size} ECGs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"loss_curve_size_{size}.png"))
    plt.close()

# ================================
# 6 Plotting the Learning Curve
# ================================
df_results = pd.DataFrame(results_list)
df_results.to_csv(os.path.join(OUTPUT_DIR, "learning_curve_results.csv"), index=False)

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Number of Training ECGs', fontsize=12)
ax1.set_ylabel('Validation R2', color='tab:blue', fontsize=12)
ax1.plot(df_results['Train_Size'], df_results['Val_R2'], marker='o', color='tab:blue', linewidth=2, label='Val R2')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()  
ax2.set_ylabel('Validation RMSE', color='tab:red', fontsize=12)  
ax2.plot(df_results['Train_Size'], df_results['Val_RMSE'], marker='s', color='tab:red', linewidth=2, linestyle='dashed', label='Val RMSE')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Using a log scale for the X-axis makes learning curves much easier to read!
plt.xscale('log') 
plt.xticks(TRAIN_SIZES, labels=[str(s) for s in TRAIN_SIZES])

plt.title('Data Efficiency: How many ECGs are needed?', fontsize=14)
fig.tight_layout()  
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "learning_curve.png"))
print(f"\nExperiment Complete! Plot saved to {OUTPUT_DIR}/learning_curve.png")