import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.patheffects as path_effects

# ==========================================
# 1. Configuration
# ==========================================
# Path to the CSV file
CSV_PATH = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/model_development/results/full_dataset_results.csv"

# Where to save the plots
OUTPUT_DIR = os.path.join(os.path.dirname(CSV_PATH), "Optuna_Visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Updated to match the cross-validation output
TARGET_METRIC = "Avg_Val_RMSE"  

# List of hyperparameter columns you want to analyze
HYPERPARAMETERS = [
    "latent_dim", "learning_rate", "base_filters", "kernel_size", 
    "num_layers", "stride_size", "pooling_type", "activation", 
    "norm_type", "dropout_rate", "batch_size", "loss_func"
]

# ==========================================
# 2. Data Loading & Cleaning
# ==========================================
print(f"Loading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH, sep=';')

if TARGET_METRIC not in df.columns:
    raise ValueError(f"Could not find '{TARGET_METRIC}' in columns: {df.columns.tolist()}")

# Filter out rows where the target metric is missing or the trial was pruned
df = df.dropna(subset=[TARGET_METRIC])
if "Pruned" in df.columns:
    df = df[df["Pruned"] == False]

# Keep only the columns that actually exist in the CSV
HYPERPARAMETERS = [col for col in HYPERPARAMETERS if col in df.columns]

print(f"Loaded {len(df)} successful trials. Analyzing against '{TARGET_METRIC}'...")

# Encode categorical variables to numbers for Math/Correlation purposes (Plots 1-5)
df_encoded = df.copy()
label_encoders = {}
for col in HYPERPARAMETERS:
    try:
        df_encoded[col] = pd.to_numeric(df_encoded[col])
    except ValueError:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

# ==========================================
# 3. Plot 1: Hyperparameter Importance (Random Forest)
# ==========================================
print("Generating Hyperparameter Importance Plot...")
X = df_encoded[HYPERPARAMETERS]
y = df_encoded[TARGET_METRIC]

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

importance_df = pd.DataFrame({
    'Hyperparameter': HYPERPARAMETERS,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
# Fixed Seaborn warning by adding hue and legend=False
sns.barplot(x='Importance', y='Hyperparameter', data=importance_df, palette='viridis', hue='Hyperparameter', legend=False)
plt.title(f'Hyperparameter Importance for predicting {TARGET_METRIC}', fontsize=14)
plt.xlabel('Importance (Relative Impact)', fontsize=12)
plt.ylabel('Hyperparameter', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_Hyperparameter_Importance.png"), dpi=300)
plt.close()

top_1_param = importance_df.iloc[0]['Hyperparameter']
top_2_param = importance_df.iloc[1]['Hyperparameter']

# ==========================================
# 4. Plot 2: Correlation Heatmap
# ==========================================
print("Generating Correlation Heatmap...")
corr_cols = HYPERPARAMETERS + [TARGET_METRIC]
corr_matrix = df_encoded[corr_cols].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
            vmax=1, vmin=-1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title(f"Correlation Heatmap (Numeric & Encoded)\nNegative correlation with {TARGET_METRIC} is GOOD (Lowers Error)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_Correlation_Heatmap.png"), dpi=300)
plt.close()

# ==========================================
# 5. Plot 3: Slice Plots (Boxplots)
# ==========================================
print("Generating Slice Plots (Boxplots) for ALL parameters...")

all_params = HYPERPARAMETERS
cols = 4
rows = math.ceil(len(all_params) / cols)

fig, axes = plt.subplots(rows, cols, figsize=(24, 6 * rows))
axes = axes.flatten()

for i, param in enumerate(all_params):
    # 1. Draw the Boxplot
    sns.boxplot(
        x=param, 
        y=TARGET_METRIC, 
        data=df, 
        ax=axes[i], 
        palette='viridis', 
        hue=param, 
        legend=False,
        showfliers=True,  # Shows outliers as dots
        width=0.6
    )
    # 2. Overlay the raw data points with jitter for better visibility
    sns.stripplot(
        x=param, 
        y=TARGET_METRIC, 
        data=df, 
        ax=axes[i], 
        color='black', 
        alpha=0.3, 
        jitter=True, 
        size=3
    )
    
    axes[i].set_title(f'Distribution of {param} vs {TARGET_METRIC}', fontsize=12, weight='bold')
    axes[i].set_xlabel(param, fontsize=11)
    axes[i].set_ylabel(TARGET_METRIC, fontsize=11)
    axes[i].grid(True, axis='y', alpha=0.3)
    
    # Rotate x-labels for readability
    if len(df[param].unique()) > 4:
        axes[i].tick_params(axis='x', rotation=45)

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_Slice_Boxplots.png"), dpi=300)
plt.close()

# ==========================================
# 5.5. Plot 3.5: Slice Plots (Marginal Effects)
# ==========================================
print("Generating Slice Plots (Marginal Effects) for ALL parameters...")

# Use all hyperparameters instead of just the top 6
all_params = HYPERPARAMETERS

# Dynamically calculate grid size (4 columns wide)
cols = 4
rows = math.ceil(len(all_params) / cols)

# Make the figure taller based on the number of rows
fig, axes = plt.subplots(rows, cols, figsize=(24, 6 * rows))
axes = axes.flatten()

for i, param in enumerate(all_params):
    # Fixed Seaborn warning by adding hue and legend=False
    sns.stripplot(x=param, y=TARGET_METRIC, data=df, ax=axes[i], alpha=0.6, jitter=True, palette='Set2', hue=param, legend=False)
    sns.pointplot(x=param, y=TARGET_METRIC, data=df, ax=axes[i], color='red', markers="D", linestyles="--", errorbar=None)
    
    axes[i].set_title(f'Effect of {param} on {TARGET_METRIC}', fontsize=12, weight='bold')
    axes[i].set_xlabel(param, fontsize=11)
    axes[i].set_ylabel(TARGET_METRIC, fontsize=11)
    axes[i].grid(True, alpha=0.3)
    
    # Rotate x-labels if there are many categories so they don't overlap
    if len(df[param].unique()) > 4:
        axes[i].tick_params(axis='x', rotation=45)

# Remove any empty subplots if the grid isn't perfectly filled
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_Slice_Plots.png"), dpi=300)
plt.close()


# ==========================================
# 6. Plot 4: Contour / Interaction Plot
# ==========================================
print(f"Generating Interaction Contour Plot for {top_1_param} vs {top_2_param}...")

plt.figure(figsize=(8, 6))
sns.kdeplot(
    data=df_encoded, x=top_1_param, y=top_2_param, 
    weights=1.0 / (df_encoded[TARGET_METRIC] + 1e-8), 
    fill=True, cmap="mako", thresh=0.05
)

sns.scatterplot(
    data=df_encoded, x=top_1_param, y=top_2_param, 
    hue=TARGET_METRIC, palette="coolwarm", s=100, edgecolor="black"
)

plt.title(f"Interaction Plot: {top_1_param} vs {top_2_param}\n(Darker/Blue dots = Better Validation RMSE)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_Interaction_Contour.png"), dpi=300)
plt.close()

# ==========================================
# 7. Plot 5: Parallel Coordinates Plot
# ==========================================
print("Generating Parallel Coordinates Plot...")
df_norm = df_encoded[corr_cols].copy()
for col in df_norm.columns:
    df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min() + 1e-8)

plt.figure(figsize=(14, 7))
df_norm = df_norm.sort_values(by=TARGET_METRIC, ascending=False)

for index, row in df_norm.iterrows():
    val = row[TARGET_METRIC]
    color = plt.cm.coolwarm(val)
    plt.plot(range(len(corr_cols)), row, color=color, alpha=0.6, linewidth=2)

plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha='right', fontsize=11)
plt.ylabel("Normalized Value (0 to 1)", fontsize=12)
plt.title("Parallel Coordinates of All Trials\n(Blue lines indicate lower Val RMSE)", fontsize=15)
plt.grid(axis='x', alpha=0.5, linestyle='--')

sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=df[TARGET_METRIC].min(), vmax=df[TARGET_METRIC].max()))
cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.02)
cbar.set_label(f"Actual {TARGET_METRIC}", rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_Parallel_Coordinates.png"), dpi=300)
plt.close()

# ==========================================
# 8. Plot 6: Quantile Distribution (Stacked Bar Plots)
# ==========================================
print("Generating Quantile Stacked Bar Plots with Outlined Inside Labels (Counts)...")

# Step 1: Sort by RMSE ascending (Lowest error = Top Tier)
df_sorted = df.sort_values(by=TARGET_METRIC, ascending=True).copy()

# Step 2: Slice into 5 equal percentage tiers SAFELY (breaking ties using rank)
tier_labels = ['1. Top 20%', '2. 20-40%', '3. 40-60%', '4. 60-80%', '5. Bottom 20%']
df_sorted['Performance_Tier'] = pd.qcut(df_sorted[TARGET_METRIC].rank(method='first'), q=5, labels=tier_labels)

# 4 columns instead of 3 for a wider grid
n_params = len(HYPERPARAMETERS)
cols = 4 
rows = math.ceil(n_params / cols)

# Made the figure slightly wider to accommodate 4 columns
fig, axes = plt.subplots(rows, cols, figsize=(24, 5 * rows))
axes = axes.flatten()

for i, param in enumerate(HYPERPARAMETERS):
    # Removed normalize='index' to keep the raw counts
    cross_tab = pd.crosstab(df_sorted['Performance_Tier'], df_sorted[param])
    
    # Plot stacked bar chart
    cross_tab.plot(kind='bar', stacked=True, ax=axes[i], colormap='viridis', edgecolor='white', alpha=0.85)
    
    # Add labels INSIDE the stacked bars
    for container in axes[i].containers:
        param_value = container.get_label()
        
        # v.get_height() is now the raw count. We draw the label if count > 0
        labels = [f"{param_value}" if v.get_height() > 0 else "" for v in container]
        
        # Add the text and capture the text objects
        text_elements = axes[i].bar_label(container, labels=labels, label_type='center', color='white', fontsize=11, weight='bold')
        
        # Add a solid black outline to every piece of text
        for text in text_elements:
            text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='black')])

    axes[i].set_title(f'{param} Distribution by Performance', fontsize=13, weight='bold')
    # Updated Y-axis label 
    axes[i].set_ylabel('Number of Trials (Count)', fontsize=11)
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=30)
    
    # Move legend outside the plot box
    axes[i].legend(title=param, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    axes[i].grid(axis='y', alpha=0.3, linestyle='--')

# Remove any empty subplots if the grid isn't perfectly filled
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "06_Stacked_Quantile_Distributions.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nSUCCESS! All 6 plots have been saved to: {OUTPUT_DIR}")