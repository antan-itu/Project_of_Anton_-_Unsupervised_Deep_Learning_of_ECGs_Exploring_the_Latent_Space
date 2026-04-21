import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
import plotly.express as px
import h5py

# --- 1. SET DIRECTORIES ---
RUN_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/model_development/experiments/GridRun_003_1804_1354"
FILE_PATH = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/data/MIMIC_IV_ECG_HDF5/mimic_iv_train.h5"

PLOT_DIR = os.path.join(RUN_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# --- 2. DEFINE TARGET LABELS ---
EXACT_LABELS = {
    "AFib": [
        "ATRIAL FIBRILLATION",
        "Atrial fibrillation",
        "Atrial fibrillation."
    ],
}

# --- 3. UMAP PARAMETERS ---
N_NEIGHBORS = 25
MIN_DIST = 0.01

# --- 4. LOAD EXPORTED VECTORS ---
print("Loading exported latent coordinates...")
try:
    latents = np.load(os.path.join(RUN_DIR, "saved_latents.npy"))
    val_idx = np.load(os.path.join(RUN_DIR, "saved_val_idx.npy"))
except FileNotFoundError as e:
    print(f"Error: Could not find necessary array files in {RUN_DIR}")
    raise e

print("Calculating 3D UMAP Projection...")
reducer_3d = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, n_components=3, random_state=42)
umap_embeddings_3d = reducer_3d.fit_transform(latents)

# --- 5. EXTRACT CLINICAL TEXT & HOVER DATA ---
print("Extracting clinical labels from GT for validation subset...")

df_gt_dict = {}
with h5py.File(FILE_PATH, 'r') as f:
    gt_group = f['GT']
    report_cols = [key for key in gt_group.keys() if key.startswith('report_')]
    for col in report_cols:
        df_gt_dict[col] = [val.decode('utf-8') for val in gt_group[col][:]]

df_gt = pd.DataFrame(df_gt_dict)
df_val_gt = df_gt.iloc[val_idx].copy()

combined_reports = df_val_gt[report_cols].fillna('').astype(str).agg(' '.join, axis=1)
clean_reports = combined_reports.str.strip().str.replace(r'\s+', ' ', regex=True)
hover_snippets = clean_reports.str.slice(0, 250) + "..."

# --- 6. LOOP THROUGH EVERY PATTERN ---
for label_name, target_list in EXACT_LABELS.items():
    print(f"\n--- Processing Label: {label_name} ---")
    
    mask = pd.Series(False, index=df_val_gt.index)
    for col in report_cols:
        if col in df_val_gt.columns:
            col_cleaned = df_val_gt[col].fillna('').astype(str).str.strip()
            mask |= col_cleaned.isin(target_list)
    
    mask = mask.astype(int)
    label_strings = [label_name if val == 1 else 'Other' for val in mask.values]
    
   
    plot_df = pd.DataFrame({
        'UMAP_3D_1': umap_embeddings_3d[:, 0],
        'UMAP_3D_2': umap_embeddings_3d[:, 1],
        'UMAP_3D_3': umap_embeddings_3d[:, 2],
        'Diagnosis': label_strings,
        'Report_Snippet': hover_snippets 
    })
    
    plot_df = plot_df.sort_values(by='Diagnosis', ascending=False)
    
    # -------------------------------------
    # Generate 36 Rotation Frames for LaTeX / Overleaf
    # -------------------------------------
    print(f"Generating 3D rotation frames for Overleaf ({label_name})...")
    frames_dir = os.path.join(PLOT_DIR, f"{label_name}_3d_frames")
    os.makedirs(frames_dir, exist_ok=True)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Separate data to plot 'Other' first, so 'AFib' renders on top
    other_mask = plot_df['Diagnosis'] == 'Other'
    target_mask = plot_df['Diagnosis'] == label_name

    # Plot 'Other'
    ax.scatter(plot_df.loc[other_mask, 'UMAP_3D_1'], 
               plot_df.loc[other_mask, 'UMAP_3D_2'], 
               plot_df.loc[other_mask, 'UMAP_3D_3'], 
               c='lightgrey', s=5, alpha=0.3, edgecolors='none', label='Other')
    
    # Plot target (AFib)
    ax.scatter(plot_df.loc[target_mask, 'UMAP_3D_1'], 
               plot_df.loc[target_mask, 'UMAP_3D_2'], 
               plot_df.loc[target_mask, 'UMAP_3D_3'], 
               c='red', s=10, alpha=0.8, edgecolors='none', label=label_name)

    # Clean up axes for a professional look
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_title(f"3D Latent Space Projection: {label_name}")
    ax.legend(loc='upper right')

    # Rotate the camera in 10-degree increments and save a frame
    for angle in range(0, 360, 10):
        ax.view_init(elev=20, azim=angle)
        # Using zero-padding (e.g., frame_000.png, frame_010.png) so LaTeX sorts them correctly
        frame_path = os.path.join(frames_dir, f"frame_{angle:03d}.png")
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')

    plt.close(fig)
    print(f"Saved 36 animation frames to: {frames_dir}")

print("\nAll patterns processed successfully!")