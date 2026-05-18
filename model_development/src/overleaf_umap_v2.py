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
BASE_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space"
RUN_DIR = os.path.join(BASE_DIR, "model_development/experiments/GridRun_001_2804_1735")
TRAIN_FILE_PATH = os.path.join(BASE_DIR, "data/MIMIC_IV_ECG_HDF5/mimic_iv_train.h5")
HOLDOUT_FILE_PATH = os.path.join(BASE_DIR, "data/MIMIC_IV_ECG_HDF5/mimic_iv_holdout.h5")
CSV_PATH = os.path.join(BASE_DIR, "data/unzipped/MIMIC_IV_ECG_CSV_MICROVOLTS_v3/record_list.csv")

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

# --- 4. LOAD EXPORTED VECTORS & APPLY CLEAN SUBSET ---
print("Loading exported holdout latents and labels...")
try:
    latents = np.load(os.path.join(RUN_DIR, "holdout_latents.npy"))
    holdout_labels = np.load(os.path.join(RUN_DIR, "holdout_labels.npy"))
except FileNotFoundError as e:
    print(f"Error: Could not find necessary array files in {RUN_DIR}")
    raise e

print("Generating pure clean holdout subset mask via record_list.csv...")
df_records = pd.read_csv(CSV_PATH)
study_to_subject = dict(zip(df_records['study_id'].astype(str), df_records['subject_id'].astype(str)))

with h5py.File(TRAIN_FILE_PATH, 'r') as f_train, h5py.File(HOLDOUT_FILE_PATH, 'r') as f_holdout:
    train_studies = [val.decode('utf-8') if isinstance(val, bytes) else str(val) for val in f_train['GT']['study_id'][:]]
    holdout_studies = [val.decode('utf-8') if isinstance(val, bytes) else str(val) for val in f_holdout['GT']['study_id'][:]]

# Map indices to subject IDs
train_patients = set([study_to_subject.get(s, f"UNMAPPED_{s}") for s in train_studies])
holdout_patients = [study_to_subject.get(s, f"UNMAPPED_{s}") for s in holdout_studies]

# Create safe mask: Keep holdout subjects ONLY if they do not exist in the training set
safe_mask = np.array([p not in train_patients for p in holdout_patients])
print(f"Mask generation complete. Found {np.sum(safe_mask)} clean ECGs out of {len(safe_mask)} total holdout ECGs.")

# Apply the clean subset mask
latents = latents[safe_mask]
holdout_labels = holdout_labels[safe_mask]

print("Calculating 3D UMAP Projection...")
reducer_3d = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, n_components=3, random_state=42)
umap_embeddings_3d = reducer_3d.fit_transform(latents)

# --- 5. EXTRACT CLINICAL TEXT & HOVER DATA ---
print("Extracting clinical labels from GT for holdout subset...")

df_gt_dict = {}
with h5py.File(HOLDOUT_FILE_PATH, 'r') as f:
    gt_group = f['GT']
    report_cols = [key for key in gt_group.keys() if key.startswith('report_')]
    for col in report_cols:
        df_gt_dict[col] = [val.decode('utf-8') for val in gt_group[col][:]]

df_gt = pd.DataFrame(df_gt_dict)

# Filter by the safe holdout subset mask
df_clean_gt = df_gt.iloc[safe_mask].copy().reset_index(drop=True)

combined_reports = df_clean_gt[report_cols].fillna('').astype(str).agg(' '.join, axis=1)
clean_reports = combined_reports.str.strip().str.replace(r'\s+', ' ', regex=True)
hover_snippets = clean_reports.str.slice(0, 250) + "..."

# --- 6. LOOP THROUGH EVERY PATTERN ---
for label_name, target_list in EXACT_LABELS.items():
    print(f"\n--- Processing Label: {label_name} ---")
    
    mask = pd.Series(False, index=df_clean_gt.index)
    for col in report_cols:
        if col in df_clean_gt.columns:
            col_cleaned = df_clean_gt[col].fillna('').astype(str).str.strip()
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
    
    plot_df['sort_order'] = plot_df['Diagnosis'].map({'Other': 0, label_name: 1})
    
    # -------------------------------------
    # DOWNSAMPLING LOGIC
    # -------------------------------------
    print(f"Downsampling 'Other' class to reduce occlusion for {label_name}...")
    df_other = plot_df[plot_df['Diagnosis'] == 'Other']
    df_target = plot_df[plot_df['Diagnosis'] == label_name]

    df_other_downsampled = df_other.sample(frac=0.30, random_state=42)

    # Recombine and sort so the target class is at the bottom of the dataframe
    plot_df_clean = pd.concat([df_other_downsampled, df_target])
    plot_df_clean = plot_df_clean.sort_values(by='sort_order')

    # Update masks for the plotting loop based on the cleaned dataframe
    other_mask = plot_df_clean['Diagnosis'] == 'Other'
    target_mask = plot_df_clean['Diagnosis'] == label_name
    
    # -------------------------------------
    # Generate Rotation Frames for LaTeX / Overleaf
    # -------------------------------------
    print(f"Generating 3D rotation frames for Overleaf ({label_name})...")
    frames_dir = os.path.join(PLOT_DIR, f"{label_name}_3d_frames")
    os.makedirs(frames_dir, exist_ok=True)

    color_map = {'Other': '#8da0cb', label_name: '#fc8d62'} 
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=k,
               markerfacecolor=v, markersize=10, markeredgecolor='k')
        for k, v in color_map.items()
    ]

    # Create 180 frames based on the 2-degree interval
    angles = range(0, 360, 2)
    print(f"Generating {len(angles)} frames in '{frames_dir}'...")

    for i, angle in enumerate(angles):
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        # Plot 'Other' - using the downsampled dataframe
        ax.scatter(
            plot_df_clean.loc[other_mask, 'UMAP_3D_1'], 
            plot_df_clean.loc[other_mask, 'UMAP_3D_2'], 
            plot_df_clean.loc[other_mask, 'UMAP_3D_3'], 
            c=color_map['Other'], s=40, alpha=0.5, edgecolor='k', linewidth=0.3
        )
        
        # Plot target - using the downsampled dataframe
        ax.scatter(
            plot_df_clean.loc[target_mask, 'UMAP_3D_1'], 
            plot_df_clean.loc[target_mask, 'UMAP_3D_2'], 
            plot_df_clean.loc[target_mask, 'UMAP_3D_3'], 
            c=color_map[label_name], s=40, alpha=1.0, edgecolor='k', linewidth=0.3
        )

        # Apply target styling and dimensions
        ax.set_title("3D Latent Space Projection")
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