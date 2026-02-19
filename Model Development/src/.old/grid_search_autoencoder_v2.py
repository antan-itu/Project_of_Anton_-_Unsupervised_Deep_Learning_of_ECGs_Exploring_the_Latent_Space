# Use mf3
# ================================
# 1 Imports & Global Config
# ================================
import os
import glob
import random
import datetime
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input, mixed_precision
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import umap.umap_ as umap
import seaborn as sns
import pandas as pd

# Reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

mixed_precision.set_global_policy('mixed_float16')

# ================================
# 2 Grid Search Configuration
# ================================
TOTAL_SAMPLES = 800035 #5087
EXPECTED_SHAPE = (5000, 8) #(5000, 12)
VAL_SPLIT = 0.20
BATCH_SIZE = 512
USE_LAYERNORM = True

# --- GRID SEARCH VARIABLES ---
GRID_LATENT_DIMS = [64]        
GRID_LEARNING_RATES = [1e-2]     
GRID_BASE_FILTERS = [32]     
GRID_KERNEL_SIZES = [5]       

# Master output directory
MASTER_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/" \
             "Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/Model Development/GridSearch"
os.makedirs(MASTER_DIR, exist_ok=True)

#DATA_PATH = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/" \
#            "Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/" \
#            "Data/MIMIC-IV_Subset/*.tfrecord"

DATA_PATH = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/" \
            "Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/" \
            "Data/MIMIC-IV/mimic_chunk_*.tfrecord"

# ================================
# 3 Reusable Pipeline Functions
# ================================
feature_description = {"ecg": tf.io.FixedLenFeature([], tf.string)}

def parse_example(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    ecg = tf.io.decode_raw(parsed["ecg"], out_type=tf.float32)
    ecg = tf.reshape(ecg, EXPECTED_SHAPE)
    
    # 1. Do the normalization math in high-precision float32
    means = tf.reduce_mean(ecg, axis=0, keepdims=True)
    stds = tf.math.reduce_std(ecg, axis=0, keepdims=True)
    ecg = (ecg - means) / (stds + 1e-8)
    
    # --- NEW: Cast to float16 to cut RAM usage by 50%! ---
    ecg = tf.cast(ecg, tf.float16)
    
    return ecg, ecg

def create_dataset(file_pattern, batch_size):
    files = tf.io.gfile.glob(file_pattern)
    files.sort()
    dataset = tf.data.TFRecordDataset(files) 
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=20000, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def build_autoencoder(input_shape, latent_dim, base_filters=32, kernel_size=5, use_layernorm=False):
    inputs = Input(shape=input_shape)
    
    # --- ENCODER ---
    x = layers.Conv1D(base_filters, kernel_size, strides=2, padding='same', activation='relu')(inputs)
    x = layers.LayerNormalization()(x) if use_layernorm else layers.BatchNormalization()(x)
    x = layers.Conv1D(base_filters * 2, kernel_size, strides=2, padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x) if use_layernorm else layers.BatchNormalization()(x)
    x = layers.Conv1D(base_filters * 4, kernel_size, strides=2, padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x) if use_layernorm else layers.BatchNormalization()(x)

    # --- LATENT SPACE ---
    shape_before_flatten = x.shape[1:]
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, name='latent_space')(x)

    # --- DECODER ---
    units = int(np.prod(shape_before_flatten))
    x = layers.Dense(units)(latent)
    x = layers.Reshape(shape_before_flatten)(x)
    x = layers.ReLU()(x)

    x = layers.Conv1DTranspose(base_filters * 4, kernel_size, strides=2, padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x) if use_layernorm else layers.BatchNormalization()(x)
    x = layers.Conv1DTranspose(base_filters * 2, kernel_size, strides=2, padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x) if use_layernorm else layers.BatchNormalization()(x)
    x = layers.Conv1DTranspose(base_filters, kernel_size, strides=2, padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x) if use_layernorm else layers.BatchNormalization()(x)

    outputs = layers.Conv1D(input_shape[-1], 1, padding='same', activation='linear')(x)
    return models.Model(inputs, outputs)

# ================================
# 4 Evaluation & Plotting Functions
# ================================
def evaluate_overall_performance(model, dataset, eval_batches, out_dir):
    print(f"Calculating final overall metrics on {eval_batches} validation batches...")
    y_true, y_pred = [], []
    for x_batch, _ in dataset.take(eval_batches):
        y_true.append(x_batch.numpy().flatten())
        y_pred.append(model.predict(x_batch, verbose=0).flatten())
        
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "Overall_MSE": float(mse),
        "Overall_RMSE": float(rmse),
        "Overall_MAE": float(mae),
        "Overall_R2": float(r2)
    }
    
    with open(os.path.join(out_dir, "final_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Final Metrics Saved: {metrics}")
    return metrics

def generate_all_plots(model, dataset, history, plot_dir, eval_batches):
    print("Generating all visualizations...")
    
    # ---------------------------------------------------------
    # Plot 1 & 2: 10 Random Reconstructions & Loss Curve
    # ---------------------------------------------------------
    for x_batch, _ in dataset.take(1):
        reconstructed = model.predict(x_batch, verbose=0)
        fig, axes = plt.subplots(5, 2, figsize=(20, 15))
        axes = axes.flatten()
        for i in range(10):
            axes[i].plot(x_batch[i, :, 0], label="Original (Lead I)", alpha=0.7)
            axes[i].plot(reconstructed[i, :, 0], label="Reconstruction", color='red', linestyle='--')
            axes[i].set_title(f"Random ECG Sample {i+1}")
            axes[i].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "01_10_random_reconstructions.png"))
        plt.close()
        break

    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history.history:
        val_loss = history.history['val_loss']
        plt.plot(epochs, val_loss, 'r--', label='Validation Loss', linewidth=2)
    plt.title('Model Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, "02_loss_curve.png"))
    plt.close()

    # ---------------------------------------------------------
    # Gather Data for Anomaly Detection & UMAP
    # ---------------------------------------------------------
    real_ecgs = []
    reconstructed_ecgs = []
    
    for x_batch, _ in dataset.take(eval_batches):
        real_ecgs.append(x_batch.numpy())
        reconstructed_ecgs.append(model.predict(x_batch, verbose=0))
        
    real_ecgs = np.concatenate(real_ecgs, axis=0)
    reconstructed_ecgs = np.concatenate(reconstructed_ecgs, axis=0)
    
    # ---------------------------------------------------------
    # Plot 3: Error Distribution Histogram
    # ---------------------------------------------------------
    mse_per_sample = np.mean(np.square(real_ecgs - reconstructed_ecgs), axis=(1, 2))
    
    plt.figure(figsize=(10, 5))
    plt.hist(mse_per_sample, bins=50, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(mse_per_sample), color='red', linestyle='dashed', linewidth=2, label='Mean Error')
    
    threshold = np.percentile(mse_per_sample, 95)
    plt.axvline(threshold, color='orange', linestyle='dashed', linewidth=2, label='95th Percentile (Anomalies)')
    
    plt.title('Distribution of Reconstruction Errors (MSE)')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Number of ECGs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, "03_error_histogram.png"))
    plt.close()
    
    # ---------------------------------------------------------
    # Plot 4: Best and Worst Reconstructions
    # ---------------------------------------------------------
    best_idx = np.argmin(mse_per_sample)  
    worst_idx = np.argmax(mse_per_sample) 
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    
    axes[0].plot(real_ecgs[best_idx, :, 0], label="Original (Lead I)", alpha=0.7)
    axes[0].plot(reconstructed_ecgs[best_idx, :, 0], label="Reconstruction", color='red', linestyle='--')
    axes[0].set_title(f"Best Reconstruction (Error: {mse_per_sample[best_idx]:.4f})")
    axes[0].legend()
    
    axes[1].plot(real_ecgs[worst_idx, :, 0], label="Original (Lead I)", alpha=0.7)
    axes[1].plot(reconstructed_ecgs[worst_idx, :, 0], label="Reconstruction", color='red', linestyle='--')
    axes[1].set_title(f"Worst Reconstruction (Error: {mse_per_sample[worst_idx]:.4f})")
    axes[1].legend()
    
    plt.savefig(os.path.join(plot_dir, "04_best_worst_reconstruction.png"))
    plt.close()

    # ---------------------------------------------------------
    # Plot 5: UMAP Visualization of Latent Space
    # ---------------------------------------------------------
    encoder = models.Model(
        inputs=model.input, 
        outputs=model.get_layer('latent_space').output
    )
    
    latent_vectors = encoder.predict(real_ecgs, verbose=0)
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_embeddings = reducer.fit_transform(latent_vectors)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=umap_embeddings[:, 0], 
        y=umap_embeddings[:, 1], 
        alpha=0.6, 
        edgecolor=None,
        s=15,
        color='b'
    )
    
    plt.title('UMAP Projection of ECG Latent Space', fontsize=14)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, "05_umap_projection.png"))
    plt.close()

# ================================
# 5 MASTER GRID SEARCH LOOP
# ================================
grid_combinations = list(itertools.product(
    GRID_LATENT_DIMS, GRID_LEARNING_RATES, GRID_BASE_FILTERS, GRID_KERNEL_SIZES
))
print(f"\nSTARTING GRID SEARCH: {len(grid_combinations)} combinations to test.\n")

val_batches = max(1, int((TOTAL_SAMPLES * VAL_SPLIT) / BATCH_SIZE))

train_dataset = create_dataset(DATA_PATH, BATCH_SIZE)
validation_dataset = train_dataset.take(val_batches)
train_dataset_final = train_dataset.skip(val_batches)

for i, (latent_dim, lr, base_filters, kernel_size) in enumerate(grid_combinations):
    
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    
    run_name = f"R{i+1}_Lat{latent_dim}_LR{lr}_F{base_filters}_K{kernel_size}_{timestamp}"
    
    run_dir = os.path.join(MASTER_DIR, run_name)
    plot_dir = os.path.join(run_dir, "plots")
    tb_log_dir = os.path.join(run_dir, "tb_logs")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    print("="*60)
    print(f"STARTING EXPERIMENT {i+1}/{len(grid_combinations)}")
    print(f"Directory: {run_name}")
    print("="*60)

    config_dict = {
        "total_samples": TOTAL_SAMPLES, "expected_shape": EXPECTED_SHAPE,
        "latent_dim": latent_dim, "batch_size": BATCH_SIZE, "learning_rate": lr,
        "base_filters": base_filters, "kernel_size": kernel_size,
        "use_layernorm": USE_LAYERNORM, "val_split": VAL_SPLIT, "val_batches": val_batches,
        "timestamp": timestamp
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    autoencoder = build_autoencoder(
        EXPECTED_SHAPE, latent_dim, 
        base_filters=base_filters, kernel_size=kernel_size, 
        use_layernorm=USE_LAYERNORM
    )
    
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')],
        jit_compile=False
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "training_history.csv"), append=True)
    
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=os.path.join(run_dir, "best_model.keras"),
    #     monitor='val_loss', save_best_only=True, verbose=0
    # )
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1)

    history = autoencoder.fit(
        train_dataset_final,
        validation_data=validation_dataset,
        epochs=500,
        callbacks=[early_stopping, csv_logger, tensorboard_cb],
        verbose=2
    )

    # Capture the metrics
    eval_batches = min(100, val_batches)
    metrics = evaluate_overall_performance(autoencoder, validation_dataset, eval_batches, out_dir=run_dir)
    generate_all_plots(autoencoder, validation_dataset, history, plot_dir, eval_batches)

    # Merge the config parameters and the final metrics into one dictionary
    summary_row = {**config_dict, **metrics}
    summary_df = pd.DataFrame([summary_row])
    
    # Save it to the main GridSearch folder
    master_csv_path = os.path.join(MASTER_DIR, "master_summary.csv")
    
    # Append the row. It will only write the header row on the very first loop
    summary_df.to_csv(master_csv_path, mode='a', index=False, header=not os.path.exists(master_csv_path))

    tf.keras.backend.clear_session()
    print(f"EXPERIMENT {i+1} COMPLETE. VRAM cleared.\n")

print("\nALL GRID SEARCH EXPERIMENTS COMPLETED SUCCESSFULLY!")