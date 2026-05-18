import matplotlib.pyplot as plt
import numpy as np
import os

# Define directory
output_dir = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/model_development/results/logs"

# Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)

#Styling
labels = ['Training (T)', 'Holdout (H)', 'Clean Holdout (C)']
x = np.arange(len(labels))
width = 0.15
models = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5']
colors = ['#1a739f', '#c18a28', '#1e8a6a', '#bd6424', '#7b5b9c']

# ==========================================
# PLOT 1: PR-AUC (XGBoost) with Error Bars
# ==========================================
means_pr = [
    [37.8, 39.7, 30.4],
    [39.9, 39.3, 28.8],
    [35.2, 35.9, 28.9],
    [34.1, 33.9, 23.8],
    [40.7, 40.0, 32.8]
]

errors_pr = [
    [[1.5, 1.1, 5.6], [1.4, 1.1, 5.8]],
    [[1.5, 1.1, 5.2], [1.5, 1.1, 6.1]],
    [[1.1, 1.0, 5.4], [1.1, 1.1, 5.9]],
    [[0.9, 1.0, 5.4], [0.9, 1.0, 5.5]],
    [[1.1, 1.1, 6.0], [1.0, 1.1, 6.6]]
]

fig1, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
for i in range(len(models)):
    offset = (i - 2) * width
    bars = ax1.bar(x + offset, means_pr[i], width, label=models[i], color=colors[i],
                  yerr=errors_pr[i], error_kw={'elinewidth': 2.5, 'ecolor': '#3b3b3b'})
    
    for bar, lower_err, upper_err in zip(bars, errors_pr[i][0], errors_pr[i][1]):
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height + upper_err),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

ax1.set_ylabel('PR-AUC (XGBoost) %', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=12)
ax1.set_ylim(15, 55)
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5, fontsize=10, frameon=True)

plt.tight_layout()
plot1_path = os.path.join(output_dir, 'prauc_xgboost_plot.png')
plt.savefig(plot1_path)
plt.close(fig1)
print(f"Saved PR-AUC plot to: {plot1_path}")


# ==========================================
# PLOT 2: All 6 Metrics in a Subplot Grid
# ==========================================
metrics_data = {
    'RMSE': [
        [0.333, 0.330, 0.289], [0.320, 0.313, 0.274], [0.379, 0.371, 0.330],
        [0.384, 0.377, 0.345], [0.258, 0.255, 0.222]
    ],
    'R²': [
        [0.889, 0.891, 0.916], [0.897, 0.902, 0.925], [0.856, 0.862, 0.891],
        [0.853, 0.858, 0.881], [0.933, 0.935, 0.951]
    ],
    'ROC-AUC (XGBoost)': [
        [0.926, 0.927, 0.935], [0.929, 0.929, 0.934], [0.913, 0.916, 0.926],
        [0.909, 0.910, 0.913], [0.920, 0.924, 0.941]
    ],
    'ROC-AUC (LogReg)': [
        [0.947, 0.950, 0.956], [0.947, 0.942, 0.949], [0.919, 0.919, 0.936],
        [0.906, 0.909, 0.931], [0.905, 0.907, 0.938]
    ],
    'PR-AUC (XGBoost)': [
        [0.378, 0.397, 0.304], [0.399, 0.393, 0.288], [0.352, 0.359, 0.289],
        [0.341, 0.339, 0.238], [0.407, 0.400, 0.328]
    ],
    'PR-AUC (LogReg)': [
        [0.449, 0.484, 0.492], [0.442, 0.404, 0.419], [0.313, 0.315, 0.277],
        [0.270, 0.283, 0.256], [0.253, 0.259, 0.247]
    ]
}

fig2, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)
axes = axes.flatten()

for idx, (title, data) in enumerate(metrics_data.items()):
    ax = axes[idx]
    for i in range(len(models)):
        offset = (i - 2) * width
        ax.bar(x + offset, data[i], width, label=models[i], color=colors[i])
    
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Train', 'Holdout', 'Clean'])
    
    if title == 'RMSE':
        ax.set_ylim(0.2, 0.45)
    elif title == 'R²':
        ax.set_ylim(0.8, 1.0)
    elif 'ROC' in title:
        ax.set_ylim(0.85, 1.0)
    elif 'PR' in title:
        ax.set_ylim(0.15, 0.55)

handles, labels_leg = ax.get_legend_handles_labels()
fig2.legend(handles, labels_leg, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=12)

plt.tight_layout()
plot2_path = os.path.join(output_dir, 'all_metrics_grid.png')
plt.savefig(plot2_path, bbox_inches='tight')
plt.close(fig2)
print(f"Saved All Metrics Grid plot to: {plot2_path}")