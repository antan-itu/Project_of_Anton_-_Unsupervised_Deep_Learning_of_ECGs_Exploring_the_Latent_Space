# ================================
# 1 Imports & Global Config
# ================================
import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input, mixed_precision
import umap.umap_ as umap
import seaborn as sns

# Reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Use mixed precision for faster training on GPUs
mixed_precision.set_global_policy('mixed_float16')

# Paths & Dataset Parameters
DATA_PATH = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/" \
            "Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/" \
            "Data/MIMIC-IV/mimic_chunk_*.tfrecord"

BASE_OUT_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/" \
               "Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/" \
               "Model Development/Logs_15_02"
PLOT_DIR = os.path.join(BASE_OUT_DIR, "plots")

# Create the folders if they don't exist
os.makedirs(BASE_OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Important: 8 Leads for MIMIC-IV, and 128 Batch Size to fit the 2080 Ti!
EXPECTED_SHAPE = (5000, 8)
BATCH_SIZE = 128
SHUFFLE_BUFFER = 20000

# ================================
# 2 TFRecord Parsing Function
# ================================
feature_description = {"ecg": tf.io.FixedLenFeature([], tf.string)}

def parse_example(example_proto):
    """Parse a single TFRecord example into (input, target) pair."""
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    ecg = tf.io.decode_raw(parsed["ecg"], out_type=tf.float32)
    ecg = tf.reshape(ecg, EXPECTED_SHAPE)
    
    # --- Z-Score Normalization per lead ---
    means = tf.reduce_mean(ecg, axis=0, keepdims=True)
    stds = tf.math.reduce_std(ecg, axis=0, keepdims=True)
    ecg = (ecg - means) / (stds + 1e-8)
    
    return ecg, ecg  # Input = Target for Autoencoder

# ================================
# 3 Dataset Creation
# ================================
def create_dataset(file_pattern=DATA_PATH, batch_size=BATCH_SIZE, shuffle_buffer=SHUFFLE_BUFFER):
    files = tf.io.gfile.glob(file_pattern)
    files.sort()
    print(f"Found {len(files)} TFRecord files.")

    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    # dataset.cache() IS REMOVED so we don't crash your server's RAM!
    dataset = dataset.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Initialize dataset
train_dataset = create_dataset()

# ================================
# 4 Autoencoder Model
# ================================
def build_autoencoder(input_shape, latent_dim=64, use_layernorm=False):
    inputs = Input(shape=input_shape)
    
    # --- ENCODER ---
    x = layers.Conv1D(32, 5, strides=2, padding='same', activation='relu')(inputs)
    x = layers.LayerNormalization()(x) if use_layernorm else layers.BatchNormalization()(x)

    x = layers.Conv1D(64, 5, strides=2, padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x) if use_layernorm else layers.BatchNormalization()(x)

    x = layers.Conv1D(128, 5, strides=2, padding='same', activation='relu')(x)
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

    x = layers.Conv1DTranspose(128, 5, strides=2, padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x) if use_layernorm else layers.BatchNormalization()(x)

    x = layers.Conv1DTranspose(64, 5, strides=2, padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x) if use_layernorm else layers.BatchNormalization()(x)

    x = layers.Conv1DTranspose(32, 5, strides=2, padding='same', activation='relu')(x)
    x = layers.LayerNormalization()(x) if use_layernorm else layers.BatchNormalization()(x)

    # Dynamically output 8 leads based on input shape
    outputs = layers.Conv1D(input_shape[-1], 1, padding='same', activation='linear')(x)
    return models.Model(inputs, outputs)

# Build & compile model
latent_dim = 64
autoencoder = build_autoencoder(EXPECTED_SHAPE, latent_dim, use_layernorm=True)
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse',
    jit_compile=False  # Turned OFF so the 2080 Ti doesn't freeze
)

autoencoder.summary()

# ================================
# 5 Training Setup
# ================================
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

csv_logger = tf.keras.callbacks.CSVLogger(
    os.path.join(BASE_OUT_DIR, "training_history.csv"), 
    append=True
)

VAL_BATCHES = 16
validation_dataset = train_dataset.take(VAL_BATCHES)
train_dataset_final = train_dataset.skip(VAL_BATCHES)

print(f"\nStarting training. All logs and outputs will be saved to:\n{BASE_OUT_DIR}\n")
history = autoencoder.fit(
    train_dataset_final,
    validation_data=validation_dataset,
    epochs=500,
    callbacks=[early_stopping, csv_logger],
    verbose=2
)

# Save the physical model
MODEL_PATH = os.path.join(BASE_OUT_DIR, "mimic_ecg_autoencoder.keras")
print(f"Training complete! Saving model to {MODEL_PATH}...")
autoencoder.save(MODEL_PATH)

# ================================
# 6 Plot Reconstruction of a Single ECG
# ================================
print("Generating Reconstruction Plot...")
def plot_reconstruction(dataset, model):
    for x_batch, _ in dataset.take(1):
        reconstructed = model.predict(x_batch)
        plt.figure(figsize=(15, 5))
        plt.plot(x_batch[0, :, 0], label="Original (Lead I)", alpha=0.7)
        plt.plot(reconstructed[0, :, 0], label="Reconstruction", color='red', linestyle='--')
        plt.title("ECG Autoencoder: Original vs Reconstruction")
        plt.legend()
        plt.savefig(os.path.join(PLOT_DIR, "01_reconstruction.png"))
        plt.close()
        break
plot_reconstruction(train_dataset, autoencoder)

# ================================
# 7 Training Loss Curve
# ================================
print("Generating Loss Curve...")
def plot_training_history(history):
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
    plt.savefig(os.path.join(PLOT_DIR, "02_loss_curve.png"))
    plt.close()
plot_training_history(history)

# ================================
# 8 Anomaly Detection (Error Distribution)
# ================================
print("Generating Anomaly Detection Plots...")
def analyze_reconstruction_errors(model, dataset, num_batches=16):
    real_ecgs = []
    reconstructed_ecgs = []
    
    for x_batch, _ in dataset.take(num_batches):
        real_ecgs.append(x_batch.numpy())
        reconstructed_ecgs.append(model.predict(x_batch, verbose=0))
        
    real_ecgs = np.concatenate(real_ecgs, axis=0)
    reconstructed_ecgs = np.concatenate(reconstructed_ecgs, axis=0)
    mse_per_sample = np.mean(np.square(real_ecgs - reconstructed_ecgs), axis=(1, 2))
    
    plt.figure(figsize=(10, 5))
    plt.hist(mse_per_sample, bins=50, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(mse_per_sample), color='red', linestyle='dashed', linewidth=2, label='Mean Error')
    threshold = np.percentile(mse_per_sample, 95)
    plt.axvline(threshold, color='orange', linestyle='dashed', linewidth=2, label='95th Percentile')
    plt.title('Distribution of Reconstruction Errors (MSE)')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Number of ECGs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, "03_error_histogram.png"))
    plt.close()
    
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
    
    plt.savefig(os.path.join(PLOT_DIR, "04_best_worst_reconstruction.png"))
    plt.close()
analyze_reconstruction_errors(autoencoder, validation_dataset, num_batches=VAL_BATCHES)

# ================================
# 9 UMAP Visualization of Latent Space
# ================================
print("Generating UMAP Visualization...")
def visualize_latent_space(model, dataset, num_batches=10):
    encoder = models.Model(inputs=model.input, outputs=model.get_layer('latent_space').output)
    ecg_samples = []
    for x_batch, _ in dataset.take(num_batches):
        ecg_samples.append(x_batch.numpy())
    ecg_samples = np.concatenate(ecg_samples, axis=0)
    
    latent_vectors = encoder.predict(ecg_samples, verbose=1)
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
    plt.savefig(os.path.join(PLOT_DIR, "05_umap_projection.png"))
    plt.close()
visualize_latent_space(autoencoder, validation_dataset, num_batches=16)

print("\nModel and plots have been successfully saved to the Logs folder.")