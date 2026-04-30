import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    "AF": [
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
    
    # Sort to ensure AFib renders on top
    plot_df['sort_order'] = plot_df['Diagnosis'].map({'Other': 0, label_name: 1})
    plot_df = plot_df.sort_values(by='sort_order')
    
    # -------------------------------------
    # Generate Rotation Frames for LaTeX / Overleaf
    # -------------------------------------
    print(f"Generating 3D rotation frames for Overleaf ({label_name})...")
    frames_dir = os.path.join(PLOT_DIR, f"{label_name}_3d_frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Setup Colors and Legend Handles
    color_map = {'Other': '#8da0cb', label_name: '#fc8d62'}
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=k,
               markerfacecolor=v, markersize=10, markeredgecolor='k')
        for k, v in color_map.items()
    ]

    other_mask = plot_df['Diagnosis'] == 'Other'
    target_mask = plot_df['Diagnosis'] == label_name

    # Create 180 frames based on the 2-degree interval
    angles = range(0, 360, 2)
    print(f"Generating {len(angles)} frames in '{frames_dir}'...")

    for i, angle in enumerate(angles):
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        # Plot 'Other'
        ax.scatter(
            plot_df.loc[other_mask, 'UMAP_3D_1'], 
            plot_df.loc[other_mask, 'UMAP_3D_2'], 
            plot_df.loc[other_mask, 'UMAP_3D_3'], 
            c=color_map['Other'], s=15, alpha=0.5, edgecolor='k', linewidth=0.2
        )
        
        # Plot target 
        ax.scatter(
            plot_df.loc[target_mask, 'UMAP_3D_1'], 
            plot_df.loc[target_mask, 'UMAP_3D_2'], 
            plot_df.loc[target_mask, 'UMAP_3D_3'], 
            c=color_map[label_name], s=40, alpha=1.0, edgecolor='k', linewidth=0.3
        )

        # Apply target styling and dimensions
        ax.set_title(f"3D Latent Space Projection")
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_zlabel('Dim 3')
        
        # Hide raw tick numbers for a cleaner topological view
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Legend
        ax.legend(handles=legend_elements, loc='upper right', title="Diagnosis")

        # Rotating Camera
        ax.view_init(elev=30, azim=angle)
        fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.1)

        # Use zero-padded frame naming for standard sorting
        frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
        plt.savefig(frame_path)
        plt.close(fig)

    print(f"Saved {len(angles)} animation frames to: {frames_dir}")

print("\nAll patterns processed successfully!")