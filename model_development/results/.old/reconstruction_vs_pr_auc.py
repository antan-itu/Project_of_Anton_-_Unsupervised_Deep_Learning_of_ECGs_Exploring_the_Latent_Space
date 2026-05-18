import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 1. Prepare the Data from the LaTeX table
data = {
    'Model': ['M1', 'M1', 'M1', 'M2', 'M2', 'M2', 'M3', 'M3', 'M3', 'M4', 'M4', 'M4', 'M5', 'M5', 'M5'],
    'Split': ['Training (T)', 'Holdout (H)', 'Clean (C)', 
              'Training (T)', 'Holdout (H)', 'Clean (C)', 
              'Training (T)', 'Holdout (H)', 'Clean (C)', 
              'Training (T)', 'Holdout (H)', 'Clean (C)', 
              'Training (T)', 'Holdout (H)', 'Clean (C)'],
    'RMSE': [0.333, 0.330, 0.289, 
             0.320, 0.313, 0.274, 
             0.379, 0.371, 0.330, 
             0.384, 0.377, 0.345, 
             0.258, 0.255, 0.222],
    'LR_PRAUC': [0.449, 0.484, 0.488, 
                 0.442, 0.404, 0.419, 
                 0.313, 0.315, 0.277, 
                 0.270, 0.283, 0.256, 
                 0.253, 0.259, 0.247],
    # Confidence Intervals for RMSE (X-axis)
    'RMSE_lower': [0.331, 0.329, 0.288, 0.316, 0.313, 0.272, 0.368, 0.370, 0.328, 0.379, 0.377, 0.344, 0.257, 0.254, 0.220],
    'RMSE_upper': [0.336, 0.330, 0.291, 0.325, 0.314, 0.276, 0.390, 0.372, 0.332, 0.389, 0.378, 0.347, 0.260, 0.256, 0.224],
    # Confidence Intervals for LogReg PR-AUC (Y-axis)
    'LR_PRAUC_lower': [0.405, 0.472, 0.419, 0.418, 0.394, 0.359, 0.295, 0.306, 0.230, 0.242, 0.276, 0.209, 0.247, 0.252, 0.205],
    'LR_PRAUC_upper': [0.494, 0.496, 0.562, 0.466, 0.415, 0.481, 0.331, 0.323, 0.328, 0.298, 0.291, 0.307, 0.258, 0.265, 0.292],
}

df_all = pd.DataFrame(data)

# 2. Global Plot Settings
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300
})

save_dir = "model_development/results/recon_vs_auc"
os.makedirs(save_dir, exist_ok=True)

# 3. Calculate Axis Limits
global_x_min = df_all['RMSE_lower'].min() - 0.02
global_x_max = df_all['RMSE_upper'].max() + 0.02
global_y_min = df_all['LR_PRAUC_lower'].min() - 0.05
global_y_max = df_all['LR_PRAUC_upper'].max() + 0.05

# 4. Loop through each dataset split
splits = df_all['Split'].unique()

for split_name in splits:
    df_split = df_all[df_all['Split'] == split_name].copy()
    
    # Calculate error bar magnitudes
    xerr = [
        df_split['RMSE'] - df_split['RMSE_lower'], 
        df_split['RMSE_upper'] - df_split['RMSE']
    ]
    yerr = [
        df_split['LR_PRAUC'] - df_split['LR_PRAUC_lower'], 
        df_split['LR_PRAUC_upper'] - df_split['LR_PRAUC']
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.set_style("whitegrid")
    ax.set_axisbelow(True)
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7, color='gray')
    
    # Overlay error bars
    ax.errorbar(
        df_split['RMSE'], df_split['LR_PRAUC'], 
        xerr=xerr, yerr=yerr, fmt='none', ecolor='black', 
        capsize=4, alpha=0.6, zorder=2
    )
    
    sns.scatterplot(
        data=df_split, x='RMSE', y='LR_PRAUC', hue='Model', 
        s=150, palette='Set1', ax=ax, zorder=3, edgecolor='black', linewidth=1.2
    )
    
    ax.set_title(f"{split_name} Dataset Performance\nwith 95% Confidence Intervals")
    ax.set_xlabel("Reconstruction Error (RMSE)")
    ax.set_ylabel("Clinical Utility (LogReg PR-AUC)")
    
    ax.set_xlim(global_x_min, global_x_max)
    ax.set_ylim(global_y_min, global_y_max)
    
    # Highlight the Baseline Model (M1)
    m1_rmse = df_split[df_split['Model'] == 'M1']['RMSE'].values[0]
    m1_prauc = df_split[df_split['Model'] == 'M1']['LR_PRAUC'].values[0]
    
    # Adjusted xytext
    ax.annotate('Baseline (M1)', 
                xy=(m1_rmse, m1_prauc), xytext=(m1_rmse + 0.015, m1_prauc + 0.02),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                zorder=4)
    
    # Move legend outside the plot
    ax.legend(title='Model', loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    
    safe_split_name = split_name.replace(' ', '_').replace('(', '').replace(')', '')
    save_path = os.path.join(save_dir, f"{safe_split_name}.png")
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig) 
    
    print(f"Plot successfully saved to: {save_path}")

print("Plots generated successfully.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 1. Prepare the Data from the LaTeX table
data = {
    'Model': ['M1', 'M1', 'M1', 'M2', 'M2', 'M2', 'M3', 'M3', 'M3', 'M4', 'M4', 'M4', 'M5', 'M5', 'M5'],
    'Split': ['Training (T)', 'Holdout (H)', 'Clean (C)', 
              'Training (T)', 'Holdout (H)', 'Clean (C)', 
              'Training (T)', 'Holdout (H)', 'Clean (C)', 
              'Training (T)', 'Holdout (H)', 'Clean (C)', 
              'Training (T)', 'Holdout (H)', 'Clean (C)'],
    # R^2 values extracted from the table
    'R2': [0.889, 0.891, 0.916, 
           0.897, 0.902, 0.925, 
           0.856, 0.862, 0.891, 
           0.853, 0.858, 0.881, 
           0.933, 0.935, 0.951],
    'LR_PRAUC': [0.449, 0.484, 0.488, 
                 0.442, 0.404, 0.419, 
                 0.313, 0.315, 0.277, 
                 0.270, 0.283, 0.256, 
                 0.253, 0.259, 0.247],
    # Confidence Intervals for R^2 (X-axis)
    'R2_lower': [0.887, 0.891, 0.915, 0.894, 0.901, 0.924, 0.847, 0.862, 0.890, 0.849, 0.857, 0.880, 0.932, 0.935, 0.950],
    'R2_upper': [0.891, 0.892, 0.917, 0.900, 0.902, 0.926, 0.865, 0.863, 0.892, 0.856, 0.858, 0.882, 0.935, 0.935, 0.952],
    # Confidence Intervals for LogReg PR-AUC (Y-axis)
    'LR_PRAUC_lower': [0.405, 0.472, 0.419, 0.418, 0.394, 0.359, 0.295, 0.306, 0.230, 0.242, 0.276, 0.209, 0.247, 0.252, 0.205],
    'LR_PRAUC_upper': [0.494, 0.496, 0.562, 0.466, 0.415, 0.481, 0.331, 0.323, 0.328, 0.298, 0.291, 0.307, 0.258, 0.265, 0.292],
}

df_all = pd.DataFrame(data)

# 2. Global Settings
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300
})

save_dir = "model_development/results/r2_vs_auc"
os.makedirs(save_dir, exist_ok=True)

# 3. Calculate Axis Limits 
global_x_min = df_all['R2_lower'].min() - 0.01
global_x_max = df_all['R2_upper'].max() + 0.01
global_y_min = df_all['LR_PRAUC_lower'].min() - 0.05
global_y_max = df_all['LR_PRAUC_upper'].max() + 0.05

# 4. Loop through each dataset split
splits = df_all['Split'].unique()

for split_name in splits:
    df_split = df_all[df_all['Split'] == split_name].copy()
    
    xerr = [
        df_split['R2'] - df_split['R2_lower'], 
        df_split['R2_upper'] - df_split['R2']
    ]
    yerr = [
        df_split['LR_PRAUC'] - df_split['LR_PRAUC_lower'], 
        df_split['LR_PRAUC_upper'] - df_split['LR_PRAUC']
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.set_style("whitegrid")
    ax.set_axisbelow(True)
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7, color='gray')
    
    # Overlay error bars
    ax.errorbar(
        df_split['R2'], df_split['LR_PRAUC'], 
        xerr=xerr, yerr=yerr, fmt='none', ecolor='black', 
        capsize=4, alpha=0.6, zorder=2
    )
    
    # Plot standard scatter
    sns.scatterplot(
        data=df_split, x='R2', y='LR_PRAUC', hue='Model', 
        s=150, palette='Set1', ax=ax, zorder=3, edgecolor='black', linewidth=1.2
    )
    
    ax.set_title(f"{split_name} Dataset Performance\nwith 95% Confidence Intervals")
    ax.set_xlabel("Reconstruction Explained Variance ($R^2$)") 
    ax.set_ylabel("Clinical Utility (LogReg PR-AUC)")
    

    ax.set_xlim(global_x_min, global_x_max)
    ax.set_ylim(global_y_min, global_y_max)
    
    m1_r2 = df_split[df_split['Model'] == 'M1']['R2'].values[0]
    m1_prauc = df_split[df_split['Model'] == 'M1']['LR_PRAUC'].values[0]
    
    ax.annotate('Baseline (M1)', 
                xy=(m1_r2, m1_prauc), xytext=(m1_r2 - 0.025, m1_prauc + 0.03),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                zorder=4)
    
    ax.legend(title='Model', loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    
    safe_split_name = split_name.replace(' ', '_').replace('(', '').replace(')', '')
    save_path = os.path.join(save_dir, f"{safe_split_name}.png")
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig) 
    
    print(f"Plot successfully saved to: {save_path}")

print("All grid-enabled, CI-annotated, consistent R^2 plots generated successfully.")