### This script is used for a side experiment to test how much training data is needed to achieve good reconstruction performance ###
import os 
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import math
import torch
import torch.nn as nn
import torch.optim as optim
import gc

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

# Setting the hyperparameters
SEQ_LEN = 5000
IN_CHANNELS = 8
LATENT_DIM = 512
LR = 0.0005
BASE_FILTERS = 32
KERNEL_SIZE = 9
NUM_LAYERS = 3
POOL_SIZE = 3 
POOLING_TYPE = 'max'
ACTIVATION = 'leaky-relu'
NORM_TYPE = 'batch'
DROPOUT_RATE = 0.0
BATCH_SIZE = 128
LOSS_FUNC = 'huber'
MASKING_RATIO = 0.0 

# The sizes of training data to test
TRAIN_SIZES = [50000, 30000, 15000, 6000, 3000, 2000, 1300, 800, 600, 450, 350, 200, 100, 50] 
VAL_SIZE = 1000

# Paths
BASE_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space"
OUTPUT_DIR = os.path.join(BASE_DIR, "model_development/how_much_data_is_enough")
H5_DATA_PATH = os.path.join(BASE_DIR, "data/MIMIC_IV_ECG_HDF5/mimic_iv_train.h5")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class for loading the dataset
class MIMIC:
    def __init__(self, h5_file_path, seq_len):
        print(f"Loading dataset into RAM from {h5_file_path}...")
        with h5py.File(h5_file_path, 'r') as h5f:
            self.data = torch.tensor(h5f['rhythm_filtered'][:], dtype=torch.float32).permute(0, 2, 1)[:, :, :seq_len]
        print(f"Dataset loaded - Shape: {self.data.shape}")
        
        print("Standardizing data...")
        means = self.data.mean(dim=2, keepdim=True)
        stds = self.data.std(dim=2, keepdim=True)
        self.data -= means
        self.data /= (stds + 1e-8)
        
        del means, stds
        gc.collect()
        print("Data standardized")

class DataLoader:
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

class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
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

# The autoencoder class
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


# Executing the experiment
print("\n" + "="*50)
print("Loarding the dataset and preparing for training...")
print("="*50)

full_dataset = MIMIC(h5_file_path=H5_DATA_PATH, seq_len=SEQ_LEN)
TOTAL_AVAILABLE = len(full_dataset.data)

all_indices = np.random.permutation(TOTAL_AVAILABLE)

# Global validation set (set to 1,000 samples)
val_indices = all_indices[:VAL_SIZE]
val_loader = DataLoader(full_dataset, val_indices, BATCH_SIZE, shuffle=False)

train_pool = all_indices[VAL_SIZE:]

results_list = []

for size in TRAIN_SIZES:
    print(f"\n>Training with {size} ECGs...")
    
    current_batch_size = min(BATCH_SIZE, size)
    current_train_idx = train_pool[:size]
    train_loader = DataLoader(full_dataset, current_train_idx, current_batch_size, shuffle=True)
    
    model = ConvAutoencoder(
        SEQ_LEN, IN_CHANNELS, LATENT_DIM, BASE_FILTERS, KERNEL_SIZE, 
        NUM_LAYERS, POOL_SIZE, ACTIVATION, DROPOUT_RATE, NORM_TYPE, POOLING_TYPE, MASKING_RATIO
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
            
        # Evaluate using the validation set
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
            
    eval_metrics = reconstruction_performance(model, val_loader, eval_batches=len(val_loader), prefix="")
    val_rmse = eval_metrics["RMSE"]
    val_r2 = eval_metrics["R2"]
    print(f"    Final Val RMSE: {val_rmse:.3f} | Final Val R2: {val_r2:.3f}")
    
    results_list.append({
        "Train_Size": size,
        "Val_RMSE": val_rmse,
        "Val_R2": val_r2
    })
    
    # Plot the loss curve for each size
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

# Plotting the learning curve
df_results = pd.DataFrame(results_list)
df_results.to_csv(os.path.join(OUTPUT_DIR, "learning_curve_results.csv"), index=False)

fig, ax1 = plt.subplots(figsize=(8, 6))

color_r2 = '#1F77B4'
color_rmse = '#4B8B3B'

ax1.set_xlabel('Number of Training ECGs', fontsize=15)
ax1.set_ylabel('Validation R2', color=color_r2, fontsize=15)
line1 = ax1.plot(df_results['Train_Size'], df_results['Val_R2'], marker='o', color=color_r2, linewidth=2.5, label='Val R2')
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelcolor=color_r2, labelsize=12)

ax2 = ax1.twinx() 
ax2.set_ylabel('Validation RMSE', color=color_rmse, fontsize=15) 
line2 = ax2.plot(df_results['Train_Size'], df_results['Val_RMSE'], marker='s', color=color_rmse, linewidth=2.5, linestyle='dashed', label='Val RMSE')
ax2.tick_params(axis='y', labelcolor=color_rmse, labelsize=12)

ax1.set_xscale('log')
ax1.set_xticks(TRAIN_SIZES)
ax1.set_xticklabels([str(s) for s in TRAIN_SIZES], rotation=45, fontsize=12)

plt.title('Reconstruction Performance vs. Training Size', fontsize=17)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', fontsize=12)

ax1.grid(True, alpha=0.3, linestyle='--', zorder=0)

fig.tight_layout() 
plt.savefig(os.path.join(OUTPUT_DIR, "learning_curve.png"), dpi=300)
print(f"\nExperiment complete - Plot saved to {OUTPUT_DIR}/learning_curve.png")