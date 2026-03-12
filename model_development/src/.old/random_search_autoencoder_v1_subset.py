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

# Setting seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Mixed precision for optimized performance
mixed_precision.set_global_policy('mixed_float16')

# ================================
# 2 Architecture & Hyperparameters
# ================================
TOTAL_SAMPLES = 5087 
EXPECTED_SHAPE = (5000, 12) 

# Cross Validation & Random Search Controls
K_FOLDS = 5                           # 5-Fold CV (Leaves 20% for validation each fold)
N_ITERATIONS = 10                     # How many random hyperparameter combinations to test

# Architecture (Expanded Search Space)
GRID_NUM_LAYERS = [3, 4]              # [3, 4] (Number of Conv blocks)
GRID_STRIDES = [2, 3]                 # [2, 3]
GRID_POOLING = ['stride', 'max', 'average'] # Type of downsampling
GRID_ACTIVATIONS = ['relu', 'leaky_relu'] # ['relu', 'leaky_relu']
GRID_NORMALIZATIONS = ['layer', 'batch']  # ['layer', 'batch', 'none']
GRID_DROPOUT_RATES = [0.0, 0.1, 0.2]  # [0.0, 0.1, 0.2] (Helps prevent overfitting)

# Training
GRID_BATCH_SIZES = [64, 128]          # [32, 64, 128]
GRID_LOSSES = ['mse', 'huber']        # ['mse', 'huber', 'mae']

# Current grid (Expanded for higher dimensionality)
GRID_LATENT_DIMS = [128, 256]
GRID_LEARNING_RATES = [0.001, 0.0005]
GRID_BASE_FILTERS = [64, 128]
GRID_KERNEL_SIZES = [5, 7, 9]

# Output directory
OUTPUT_DIR = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/" \
             "Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/Model Development/GridSearch"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data Path for the Subset
DATA_PATH = "/home/akokholm/mnt/SUN-BMI-EC-AKOKHOLM/Master-BMI/GitHub_Repository/" \
            "Project_of_Anton_-_Unsupervised_Deep_Learning_of_ECGs_Exploring_the_Latent_Space/" \
            "Data/MIMIC-IV_Subset/*.tfrecord"

# ================================
# 3 Pipeline Functions
# ================================
feature_description = {"ecg": tf.io.FixedLenFeature([], tf.string)}

# Function to parse TFRecords and apply Z-score normalization
def parse_example(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    ecg = tf.io.decode_raw(parsed["ecg"], out_type=tf.float32)
    ecg = tf.reshape(ecg, EXPECTED_SHAPE)
    
    # Z-Score Normalization
    means = tf.reduce_mean(ecg, axis=0, keepdims=True)    # Compute mean across time dimension
    stds = tf.math.reduce_std(ecg, axis=0, keepdims=True) # Compute std across time dimension
    ecg = (ecg - means) / (stds + 1e-8)                   # Preventing division by zero
    return ecg, ecg

# Function to create K-Fold split datasets from TFRecords
def create_kfold_datasets(file_pattern, batch_size, fold, k_folds):
    files = tf.io.gfile.glob(file_pattern)
    files.sort()
    
    # Base unbatched dataset
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Split using enumeration (i % k_folds == fold goes to validation)
    def is_val(i, x): return i % k_folds == fold
    def is_train(i, x): return i % k_folds != fold
    
    val_dataset = dataset.enumerate().filter(is_val).map(lambda i, x: x)
    train_dataset = dataset.enumerate().filter(is_train).map(lambda i, x: x)
    
    # Cache and batch independently
    train_dataset = train_dataset.cache().shuffle(buffer_size=20000).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset

# Function to build a convolutional autoencoder based on the provided hyperparameters
def build_autoencoder(input_shape, latent_dim, base_filters, kernel_size, num_layers, stride_size, activation, dropout_rate, norm_type, pooling_type):
    inputs = Input(shape=input_shape) # Input layer for the autoencoder, using the defined shape
    x = inputs
    
    # ENCODER
    for i in range(num_layers): # Loop to create the specified number of convolutional layers
        filters = base_filters * (2**i) # Increasing the number of filters with each layer
        
        # If pooling type is 'stride', use strides in Conv1D. Else use stride=1 and rely on pooling layer.
        conv_stride = stride_size if pooling_type == 'stride' else 1
        x = layers.Conv1D(filters, kernel_size, strides=conv_stride, padding='same')(x)
        
        if norm_type == 'layer': # Apply layer normalization if specified
            x = layers.LayerNormalization()(x)
        elif norm_type == 'batch': # Apply batch normalization if specified
            x = layers.BatchNormalization()(x)
            
        if activation == 'leaky_relu': # Use LeakyReLU activation if specified
            x = layers.LeakyReLU()(x)
        else:
            x = layers.Activation(activation)(x)
            
        # Apply specific pooling if requested
        if pooling_type == 'max':
            x = layers.MaxPooling1D(pool_size=stride_size, padding='same')(x)
        elif pooling_type == 'average':
            x = layers.AveragePooling1D(pool_size=stride_size, padding='same')(x)
            
        if dropout_rate > 0.0: # Apply dropout if specified - prevent overfitting
            x = layers.Dropout(dropout_rate)(x)

    # LATENT SPACE
    shape_before_flatten = x.shape[1:] # Store the shape
    x = layers.Flatten()(x) # Flatten the output to feed into the dense layer for latent space
    latent = layers.Dense(latent_dim, name='latent_space')(x) # Creating the latent space with the specified dimension

    # DECODER
    units = int(np.prod(shape_before_flatten)) # Calculate the number of units to reshape
    x = layers.Dense(units)(latent)
    x = layers.Reshape(shape_before_flatten)(x) # Reshaping
    
    if activation == 'leaky_relu':
        x = layers.LeakyReLU()(x)
    else:
        x = layers.Activation(activation)(x)

    # Reverse the filters for the decoder
    for i in reversed(range(num_layers)): # Loop to create the decoder layers in reverse order
        filters = base_filters * (2**i) # Decreasing the number of filters with each layer
        
        # Reverse the pooling with UpSampling
        if pooling_type in ['max', 'average']:
            x = layers.UpSampling1D(size=stride_size)(x)
            conv_stride = 1
        else:
            conv_stride = stride_size
            
        x = layers.Conv1DTranspose(filters, kernel_size, strides=conv_stride, padding='same')(x) # Transposing for upsampling
        
        if norm_type == 'layer':
            x = layers.LayerNormalization()(x)
        elif norm_type == 'batch':
            x = layers.BatchNormalization()(x)
            
        if activation == 'leaky_relu':
            x = layers.LeakyReLU()(x)
        else:
            x = layers.Activation(activation)(x)
            
        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate)(x)

    # Output layer - reconstructing the ECG with linear activation for regression
    x = layers.Conv1D(input_shape[-1], 1, padding='same', activation='linear')(x)
    
    # Crop the output to exactly match the input shape (fixes rounding errors from odd strides)
    outputs = layers.Lambda(lambda tensor: tensor[:, :input_shape[0], :])(x)
    
    return models.Model(inputs, outputs)

# ================================
# 4 Evaluation & Plotting Functions
# ================================
# Function to evaluate performance with a dynamic prefix (Train_ or Val_)
def evaluate_overall_performance(model, dataset, eval_batches, prefix=""):
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
        f"{prefix}MSE": float(mse),
        f"{prefix}RMSE": float(rmse),
        f"{prefix}MAE": float(mae),
        f"{prefix}R2": float(r2)
    }
    return metrics

# Function to generate plots for reconstructions, loss curves, error distributions, and UMAP projections
def generate_all_plots(model, dataset, history, plot_dir, eval_batches):
    # Plotting 10 reconstructions
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

    # Plotting loss curve
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history.history:
        plt.plot(epochs, history.history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    plt.title('Model Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, "02_loss_curve.png"))
    plt.close()

    # Plotting error distributions for the reconstructions and the best/worst reconstructions
    real_ecgs, reconstructed_ecgs = [], []
    for x_batch, _ in dataset.take(eval_batches):
        real_ecgs.append(x_batch.numpy())
        reconstructed_ecgs.append(model.predict(x_batch, verbose=0))
        
    real_ecgs = np.concatenate(real_ecgs, axis=0)
    reconstructed_ecgs = np.concatenate(reconstructed_ecgs, axis=0)
    
    mse_per_sample = np.mean(np.square(real_ecgs - reconstructed_ecgs), axis=(1, 2))
    
    plt.figure(figsize=(10, 5))
    plt.hist(mse_per_sample, bins=50, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(mse_per_sample), color='red', linestyle='dashed', linewidth=2, label='Mean Error')
    plt.title('Distribution of Reconstruction Errors (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, "03_error_histogram.png"))
    plt.close()
    
    best_idx = np.argmin(mse_per_sample)  
    worst_idx = np.argmax(mse_per_sample) 
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    axes[0].plot(real_ecgs[best_idx, :, 0], label="Original (Lead I)", alpha=0.7)
    axes[0].plot(reconstructed_ecgs[best_idx, :, 0], label="Reconstruction", color='red', linestyle='--')
    axes[0].set_title(f"Best Reconstruction (Error: {mse_per_sample[best_idx]:.4f})")
    
    axes[1].plot(real_ecgs[worst_idx, :, 0], label="Original (Lead I)", alpha=0.7)
    axes[1].plot(reconstructed_ecgs[worst_idx, :, 0], label="Reconstruction", color='red', linestyle='--')
    axes[1].set_title(f"Worst Reconstruction (Error: {mse_per_sample[worst_idx]:.4f})")
    plt.savefig(os.path.join(plot_dir, "04_best_worst_reconstruction.png"))
    plt.close()

    # Creating the UMAP projection of the latent space
    encoder = models.Model(inputs=model.input, outputs=model.get_layer('latent_space').output)
    latent_vectors = encoder.predict(real_ecgs, verbose=0)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap_embeddings = reducer.fit_transform(latent_vectors)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=umap_embeddings[:, 0], y=umap_embeddings[:, 1], alpha=0.6, s=15, color='b')
    plt.title('UMAP Projection of ECG Latent Space', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, "05_umap_projection.png"))
    plt.close()

# ================================
# 5 RANDOMIZED SEARCH & K-FOLD LOOP
# ================================
# 1. Generate ALL possible combinations
all_combinations = list(itertools.product(
    GRID_LATENT_DIMS, GRID_LEARNING_RATES, GRID_BASE_FILTERS, GRID_KERNEL_SIZES,
    GRID_NUM_LAYERS, GRID_STRIDES, GRID_ACTIVATIONS, GRID_NORMALIZATIONS, 
    GRID_DROPOUT_RATES, GRID_BATCH_SIZES, GRID_LOSSES, GRID_POOLING
))

# 2. Randomly sample N configurations
random.shuffle(all_combinations)
sampled_combinations = all_combinations[:N_ITERATIONS]

print(f"\nSTARTING RANDOM SEARCH: Testing {N_ITERATIONS} configurations with {K_FOLDS}-Fold CV.\n")

# Looping through the randomized grid
for i, (latent_dim, lr, base_filters, kernel_size, num_layers, stride_size, activation, norm_type, dropout_rate, batch_size, loss_func, pooling_type) in enumerate(sampled_combinations):
    
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M") # Timestamp for the folders
    readable_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Setting folder names and paths
    run_name = f"R{i+1}_Lat{latent_dim}_Lyr{num_layers}_Str{stride_size}_{pooling_type}_F{base_filters}_{timestamp}"
    run_dir = os.path.join(OUTPUT_DIR, run_name)
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    print("="*60)
    print(f"EXPERIMENT {i+1}/{N_ITERATIONS}: {run_name}")
    print("="*60)

    # Dictionary to hold the metrics for all K folds
    fold_metrics_list = []
    best_fold_model = None
    best_fold_history = None
    best_fold_dataset = None
    best_r2 = -float('inf')

    # --- K-FOLD CROSS VALIDATION LOOP ---
    for fold in range(K_FOLDS):
        print(f"  --> Starting Fold {fold + 1}/{K_FOLDS}")
        
        # 1. Create the unique split for this fold
        train_dataset, validation_dataset = create_kfold_datasets(DATA_PATH, batch_size, fold, K_FOLDS)
        
        # 2. Build fresh model
        autoencoder = build_autoencoder(
            EXPECTED_SHAPE, latent_dim, base_filters, kernel_size, 
            num_layers, stride_size, activation, dropout_rate, norm_type, pooling_type
        )
        
        autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=loss_func,
            jit_compile=True 
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # 3. Train
        history = autoencoder.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=500,
            callbacks=[early_stopping],
            verbose=0 # Suppress spam, we are printing fold-level progress
        )
        
        # 4. Evaluate Fold (Train and Validation Separately!)
        eval_batches = min(16, max(1, int((TOTAL_SAMPLES / K_FOLDS) / batch_size)))
        
        train_metrics = evaluate_overall_performance(autoencoder, train_dataset, eval_batches, prefix="Train_")
        val_metrics = evaluate_overall_performance(autoencoder, validation_dataset, eval_batches, prefix="Val_")
        
        # Combine train and val metrics for this fold
        fold_metrics = {**train_metrics, **val_metrics}
        fold_metrics["Fold"] = fold + 1
        fold_metrics_list.append(fold_metrics)
        
        print(f"      Fold {fold + 1} -> Train R2: {train_metrics['Train_R2']:.4f} | Val R2: {val_metrics['Val_R2']:.4f}")
        
        # Keep track of the best fold for plotting
        if val_metrics['Val_R2'] > best_r2:
            best_r2 = val_metrics['Val_R2']
            best_fold_model = autoencoder
            best_fold_history = history
            best_fold_dataset = validation_dataset

    # --- SAVE PER-FOLD METRICS ---
    with open(os.path.join(run_dir, "fold_metrics.json"), "w") as f:
        json.dump(fold_metrics_list, f, indent=4)

    # --- CALCULATE THE AVERAGES ACROSS ALL FOLDS ---
    avg_metrics = {}
    for metric_name in fold_metrics_list[0].keys():
        if metric_name != "Fold":
            avg_metrics[f"Avg_{metric_name}"] = np.mean([m[metric_name] for m in fold_metrics_list])
            avg_metrics[f"Std_{metric_name}"] = np.std([m[metric_name] for m in fold_metrics_list])

    print(f"  >>> Average K-Fold Val R2: {avg_metrics['Avg_Val_R2']:.4f} (±{avg_metrics['Std_Val_R2']:.4f})")

    # Save averaged metrics to JSON
    with open(os.path.join(run_dir, "avg_metrics.json"), "w") as f:
        json.dump(avg_metrics, f, indent=4)

    # Generate plots using the BEST performing fold so visualizations look clean
    generate_all_plots(best_fold_model, best_fold_dataset, best_fold_history, plot_dir, eval_batches)

    # Saving configuration for reproducibility
    config_dict = {
        "date": readable_date, "total_samples": TOTAL_SAMPLES, "expected_shape": str(EXPECTED_SHAPE),
        "latent_dim": latent_dim, "learning_rate": lr, "base_filters": base_filters, 
        "kernel_size": kernel_size, "num_layers": num_layers, "stride_size": stride_size,
        "pooling_type": pooling_type, "activation": activation, "norm_type": norm_type, 
        "dropout_rate": dropout_rate, "batch_size": batch_size, "loss_func": loss_func,
        "k_folds": K_FOLDS
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    # Save to Summary (Now explicitly tracking Avg_Train_R2, Avg_Val_R2, etc.)
    summary_row = {**config_dict, **avg_metrics}
    summary_df = pd.DataFrame([summary_row])
    csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    summary_df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))

    tf.keras.backend.clear_session()
    print(f"EXPERIMENT {i+1} COMPLETE\n")

print("\nALL RANDOM SEARCH EXPERIMENTS COMPLETED!")