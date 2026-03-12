import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
import plotly.express as px

# --- 1. SET DIRECTORIES ---
RUN_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/Model Development/FullGridSearch/GridRun_001_1103_1008"
FILE_PATH = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/Data/Full training dataset/training_dataset.h5"

PLOT_DIR = os.path.join(RUN_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# --- 2. DEFINE YOUR EXACT TARGET LABELS ---
EXACT_LABELS = {
    "AFib": [
        "ATRIAL FIBRILLATION",
        "Atrial fibrillation",
        "Atrial fibrillation."
    ],
}

# --- 3. TUNED UMAP PARAMETERS ---
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

print(f"Calculating 2D UMAP (Neighbors: {N_NEIGHBORS}, Min Dist: {MIN_DIST})...")
reducer_2d = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, n_components=2, random_state=42)
umap_embeddings_2d = reducer_2d.fit_transform(latents)

print("Calculating 3D UMAP Projection...")
reducer_3d = umap.UMAP(n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, n_components=3, random_state=42)
umap_embeddings_3d = reducer_3d.fit_transform(latents)

# --- 5. EXTRACT CLINICAL TEXT & HOVER DATA ---
print("Extracting clinical labels from GT for validation subset...")
df_gt = pd.read_hdf(FILE_PATH, key='GT')
report_cols = [f'report_{i}' for i in range(18)]

# Extract only the samples evaluated by the model
df_val_gt = df_gt.iloc[val_idx].copy()

# Combine the text and clean up whitespace for the tooltips
combined_reports = df_val_gt[report_cols].fillna('').astype(str).agg(' '.join, axis=1)
clean_reports = combined_reports.str.strip().str.replace(r'\s+', ' ', regex=True)

# Truncate to the first 250 characters so the Plotly hover box doesn't cover the entire screen
hover_snippets = clean_reports.str.slice(0, 250) + "..."

# --- 6. LOOP THROUGH EVERY PATTERN ---
for label_name, target_list in EXACT_LABELS.items():
    print(f"\n--- Processing Label: {label_name} ---")
    
    # Apply Strict Match Logic
    mask = pd.Series(False, index=df_val_gt.index)

    for col in report_cols:
        if col in df_val_gt.columns:
            # Clean the column exactly like we did in the GT exploration script
            col_cleaned = df_val_gt[col].fillna('').astype(str).str.strip()
            # Use strict isin() matching instead of regex contains()
            mask |= col_cleaned.isin(target_list)
    
    mask = mask.astype(int)
    label_strings = [label_name if val == 1 else 'Other' for val in mask.values]
    
    # -------------------------------------
    # Generate 2D PNG (Seaborn)
    # -------------------------------------
    print(f"Generating 2D PNG for {label_name}...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=umap_embeddings_2d[:, 0], 
        y=umap_embeddings_2d[:, 1], 
        hue=label_strings,
        palette={label_name: 'red', 'Other': 'lightgrey'},
        alpha=0.6, 
        s=15,
        edgecolor=None
    )
    plt.title(f'2D UMAP (n_neighbors={N_NEIGHBORS}): {label_name}', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path_2d = os.path.join(PLOT_DIR, f"{label_name}_umap_2d.png")
    plt.savefig(save_path_2d, dpi=300)
    plt.close() 
    
    # -------------------------------------
    # Generate 3D HTML with Hover Tooltips (Plotly)
    # -------------------------------------
    print(f"Generating Interactive 3D HTML for {label_name}...")
    plot_df = pd.DataFrame({
        'UMAP_3D_1': umap_embeddings_3d[:, 0],
        'UMAP_3D_2': umap_embeddings_3d[:, 1],
        'UMAP_3D_3': umap_embeddings_3d[:, 2],
        'Diagnosis': label_strings,
        'Report_Snippet': hover_snippets 
    })
    
    plot_df = plot_df.sort_values(by='Diagnosis', ascending=False)
    
    fig_3d = px.scatter_3d(
        plot_df, 
        x='UMAP_3D_1', 
        y='UMAP_3D_2', 
        z='UMAP_3D_3',
        color='Diagnosis',
        color_discrete_map={label_name: 'red', 'Other': 'lightgrey'},
        opacity=0.6,
        hover_data={'Report_Snippet': True, 'UMAP_3D_1': False, 'UMAP_3D_2': False, 'UMAP_3D_3': False},
        title=f'Interactive 3D UMAP (n_neighbors={N_NEIGHBORS}): {label_name} (Hover for Clinical Text)'
    )
    
    fig_3d.update_traces(marker=dict(size=3, line=dict(width=0)))
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        margin=dict(r=10, b=10, l=10, t=40)
    )
    
    save_path_3d = os.path.join(PLOT_DIR, f"{label_name}_umap_3d_interactive.html")
    fig_3d.write_html(save_path_3d)
    
    print(f"Saved: {save_path_2d}")
    print(f"Saved: {save_path_3d}")

print("\nAll patterns processed successfully!")