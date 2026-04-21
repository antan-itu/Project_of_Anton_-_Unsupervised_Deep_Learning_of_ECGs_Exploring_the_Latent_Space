import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --- 1. CONFIGURATION ---
RUN_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/model_development/experiments/GridRun_001_0804_1521"
DATA_PATH = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/data/MIMIC_IV_ECG_HDF5/mimic_iv_train.h5"
MODEL_PATH = os.path.join(RUN_DIR, "best_fold_model.pth")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 512 # Massive batch size for fast inference

# --- 2. REBUILD THE EXACT ARCHITECTURE ---
class ConvAutoencoder(nn.Module):
    # This must match your Grid Search parameters exactly
    def __init__(self, seq_len=5000, in_channels=8, latent_dim=512, base_filters=64, kernel_size=9,
                 num_layers=3, pool_size=3, activation='leaky_relu', dropout_rate=0.0, norm_type='batch', pooling_type='average'):
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
            if norm_type == 'batch': encoder_layers.append(nn.BatchNorm1d(filters))
            encoder_layers.append(nn.LeakyReLU() if activation == 'leaky_relu' else nn.ReLU())
            if pooling_type == 'average': encoder_layers.append(nn.AvgPool1d(kernel_size=pool_size, stride=pool_size))
            current_channels = filters
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        dummy_input = torch.zeros(1, in_channels, seq_len)
        dummy_output = self.encoder(dummy_input)
        self.shape_before_flatten = dummy_output.shape[1:]
        flattened_size = int(np.prod(self.shape_before_flatten))
        
        self.fc_latent = nn.Linear(flattened_size, latent_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        flattened = encoded.view(encoded.size(0), -1)
        latent = self.fc_latent(flattened)
        return latent

# --- 3. EXECUTE EXTRACTION ---
def extract_all():
    print(f"Loading Model to {DEVICE}...")
    model = ConvAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True), strict=False)
    model.eval()

    print(f"\nLoading entire HDF5 dataset into RAM...")
    with h5py.File(DATA_PATH, 'r') as h5f:
        # Load matrices
        raw_data = torch.tensor(h5f['rhythm_filtered'][:], dtype=torch.float32).permute(0, 2, 1)
        
        # Load and decode text reports safely
        gt_group = h5f['GT']
        report_cols = [key for key in gt_group.keys() if key.startswith('report_')]
        df_gt_dict = {}
        for col in report_cols:
            df_gt_dict[col] = [val.decode('utf-8') for val in gt_group[col][:]]
            
    df_gt = pd.DataFrame(df_gt_dict)
    print(f"Loaded {len(raw_data)} ECGs.")

    print("\nStandardizing data (In-Place)...")
    means = raw_data.mean(dim=2, keepdim=True)
    stds = raw_data.std(dim=2, keepdim=True)
    raw_data -= means
    raw_data /= (stds + 1e-8)

    print("\nExtracting Latent Vectors (Batch Inference)...")
    dataloader = DataLoader(raw_data, batch_size=BATCH_SIZE, shuffle=False)
    
    all_latents = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.to(DEVICE)
            latent = model(batch)
            all_latents.append(latent.cpu().numpy())
            if (i+1) % 100 == 0:
                print(f"Processed {(i+1)*BATCH_SIZE} / {len(raw_data)} ECGs...", end='\r')

    all_latents = np.concatenate(all_latents, axis=0)
    print(f"\nExtraction complete! Latent matrix shape: {all_latents.shape}")

    print("\nBuilding Exact AFib Labels...")
    EXACT_TARGETS = ["ATRIAL FIBRILLATION", "Atrial fibrillation", "Atrial fibrillation."]
    mask = pd.Series(False, index=df_gt.index)
    for col in report_cols:
        if col in df_gt.columns:
            mask |= df_gt[col].fillna('').astype(str).str.strip().isin(EXACT_TARGETS)
            
    y_labels = mask.astype(int).values
    print(f"Total AFib Cases Found: {sum(y_labels)}")

    print("\nSaving massive arrays to disk...")
    np.save(os.path.join(RUN_DIR, "FULL_latents.npy"), all_latents)
    np.save(os.path.join(RUN_DIR, "FULL_afib_labels.npy"), y_labels)
    print(f"Saved successfully to: {RUN_DIR}")

if __name__ == "__main__":
    extract_all()