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
import optuna
import gc

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

# Cross Validation & Optuna Controls
K_FOLDS = 5                           
N_ITERATIONS = 1                     

# Architecture
GRID_NUM_LAYERS = [3]              
GRID_STRIDES = [2]                 
GRID_POOLING = ['stride'] 
GRID_ACTIVATIONS = ['leaky_relu'] 
GRID_NORMALIZATIONS = ['batch']  
GRID_DROPOUT_RATES = [0.1]  

# Training
GRID_BATCH_SIZES = [64]          
GRID_LOSSES = ['mse']        

# Current grid 
GRID_LATENT_DIMS = [512]
GRID_LEARNING_RATES = [0.0005]
GRID_BASE_FILTERS = [256] #128
GRID_KERNEL_SIZES = [9]

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

def parse_example(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    ecg = tf.io.decode_raw(parsed["ecg"], out_type=tf.float32)
    ecg = tf.reshape(ecg, EXPECTED_SHAPE)
    
    means = tf.reduce_mean(ecg, axis=0, keepdims=True)    
    stds = tf.math.reduce_std(ecg, axis=0, keepdims=True) 
    ecg = (ecg - means) / (stds + 1e-8)                   
    return ecg, ecg

def create_kfold_datasets(file_pattern, batch_size, fold, k_folds):
    files = tf.io.gfile.glob(file_pattern)
    files.sort()
    
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    
    def is_val(i, x): return i % k_folds == fold
    def is_train(i, x): return i % k_folds != fold
    
    val_dataset = dataset.enumerate().filter(is_val).map(lambda i, x: x)
    train_dataset = dataset.enumerate().filter(is_train).map(lambda i, x: x)
    
    train_dataset = train_dataset.cache().shuffle(buffer_size=20000).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset

def build_autoencoder(input_shape, latent_dim, base_filters, kernel_size, num_layers, stride_size, activation, dropout_rate, norm_type, pooling_type):
    inputs = Input(shape=input_shape) 
    x = inputs
    
    # ENCODER
    for i in range(num_layers): 
        filters = base_filters * (2**i) 
        
        conv_stride = stride_size if pooling_type == 'stride' else 1
        x = layers.Conv1D(filters, kernel_size, strides=conv_stride, padding='same')(x)
        
        if norm_type == 'layer': 
            x = layers.LayerNormalization()(x)
        elif norm_type == 'batch': 
            x = layers.BatchNormalization()(x)
            
        if activation == 'leaky_relu': 
            x = layers.LeakyReLU()(x)
        else:
            x = layers.Activation(activation)(x)
            
        if pooling_type == 'max':
            x = layers.MaxPooling1D(pool_size=stride_size, padding='same')(x)
        elif pooling_type == 'average':
            x = layers.AveragePooling1D(pool_size=stride_size, padding='same')(x)
            
        if dropout_rate > 0.0: 
            x = layers.Dropout(dropout_rate)(x)

    # LATENT SPACE
    shape_before_flatten = x.shape[1:] 
    x = layers.Flatten()(x) 
    latent = layers.Dense(latent_dim, name='latent_space')(x) 

    # DECODER
    units = int(np.prod(shape_before_flatten)) 
    x = layers.Dense(units)(latent)
    x = layers.Reshape(shape_before_flatten)(x) 
    
    if activation == 'leaky_relu':
        x = layers.LeakyReLU()(x)
    else:
        x = layers.Activation(activation)(x)

    for i in reversed(range(num_layers)): 
        filters = base_filters * (2**i) 
        
        if pooling_type in ['max', 'average']:
            x = layers.UpSampling1D(size=stride_size)(x)
            conv_stride = 1
        else:
            conv_stride = stride_size
            
        x = layers.Conv1DTranspose(filters, kernel_size, strides=conv_stride, padding='same')(x) 
        
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

    x = layers.Conv1D(input_shape[-1], 1, padding='same', activation='linear')(x)
    outputs = layers.Lambda(lambda tensor: tensor[:, :input_shape[0], :])(x)
    
    return models.Model(inputs, outputs)

# ================================
# 4 Evaluation & Plotting Functions
# ================================
def evaluate_overall_performance(model, dataset, eval_batches, prefix=""):
    y_true, y_pred = [], []
    for x_batch, _ in dataset.take(eval_batches): 
        y_true.append(x_batch.numpy().flatten())
        y_pred.append(model.predict(x_batch, verbose=0).flatten())
        
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    
    mse = round(float(mean_squared_error(y_true, y_pred)), 3)
    rmse = round(float(np.sqrt(mse)), 3)
    mae = round(float(mean_absolute_error(y_true, y_pred)), 3)
    r2 = round(float(r2_score(y_true, y_pred)), 3)
    
    metrics = {
        f"{prefix}MSE": mse,
        f"{prefix}RMSE": rmse,
        f"{prefix}MAE": mae,
        f"{prefix}R2": r2
    }
    return metrics

# FIXED: Passed history_dict instead of the massive history object
def generate_all_plots(model, dataset, history_dict, plot_dir, eval_batches):
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

    # Extracting strictly the numbers!
    loss = history_dict['loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history_dict:
        plt.plot(epochs, history_dict['val_loss'], 'r--', label='Validation Loss', linewidth=2)
    plt.title('Model Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, "02_loss_curve.png"))
    plt.close()

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
# 5 OPTUNA BAYESIAN OPTIMIZATION
# ================================
print(f"\nSTARTING OPTUNA OPTIMIZATION: Testing {N_ITERATIONS} trials with {K_FOLDS}-Fold CV.\n")

def objective(trial):
    latent_dim = trial.suggest_categorical('latent_dim', GRID_LATENT_DIMS)
    lr = trial.suggest_categorical('learning_rate', GRID_LEARNING_RATES)
    base_filters = trial.suggest_categorical('base_filters', GRID_BASE_FILTERS)
    kernel_size = trial.suggest_categorical('kernel_size', GRID_KERNEL_SIZES)
    num_layers = trial.suggest_categorical('num_layers', GRID_NUM_LAYERS)
    stride_size = trial.suggest_categorical('stride_size', GRID_STRIDES)
    activation = trial.suggest_categorical('activation', GRID_ACTIVATIONS)
    norm_type = trial.suggest_categorical('norm_type', GRID_NORMALIZATIONS)
    dropout_rate = trial.suggest_categorical('dropout_rate', GRID_DROPOUT_RATES)
    batch_size = trial.suggest_categorical('batch_size', GRID_BATCH_SIZES)
    loss_func = trial.suggest_categorical('loss_func', GRID_LOSSES)
    pooling_type = trial.suggest_categorical('pooling_type', GRID_POOLING)

    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    readable_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    run_name = f"Trial{trial.number}_Lat{latent_dim}_Lyr{num_layers}_{pooling_type}_F{base_filters}_{timestamp}"
    run_dir = os.path.join(OUTPUT_DIR, run_name)
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    print("="*60)
    print(f"STARTING OPTUNA TRIAL {trial.number}: {run_name}")
    print("="*60)

    fold_metrics_list = []
    best_fold_history_dict = None # FIXED
    best_fold_dataset = None
    best_r2 = -float('inf')
    is_pruned = False 
    
    best_model_path = os.path.join(run_dir, "best_fold_model.h5")

    try:
        for fold in range(K_FOLDS):
            print(f"  --> Starting Fold {fold + 1}/{K_FOLDS}")
            
            train_dataset, validation_dataset = create_kfold_datasets(DATA_PATH, batch_size, fold, K_FOLDS)
            
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
            
            history = autoencoder.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=500,
                callbacks=[early_stopping],
                verbose=0 
            )
            
            eval_batches = min(16, max(1, int((TOTAL_SAMPLES / K_FOLDS) / batch_size)))
            train_metrics = evaluate_overall_performance(autoencoder, train_dataset, eval_batches, prefix="Train_")
            val_metrics = evaluate_overall_performance(autoencoder, validation_dataset, eval_batches, prefix="Val_")
            
            fold_metrics = {**train_metrics, **val_metrics}
            fold_metrics["Fold"] = fold + 1
            fold_metrics_list.append(fold_metrics)
            
            print(f"      Fold {fold + 1} -> Train R2: {train_metrics['Train_R2']:.3f} | Val R2: {val_metrics['Val_R2']:.3f}")
            
            if val_metrics['Val_R2'] > best_r2:
                best_r2 = val_metrics['Val_R2']
                autoencoder.save(best_model_path) 
                best_fold_history_dict = history.history 
                best_fold_dataset = validation_dataset

            trial.report(val_metrics['Val_R2'], step=fold)
            tf.keras.backend.clear_session()
            del autoencoder
            del history
            del train_dataset
            del validation_dataset
            gc.collect()

            if trial.should_prune():
                is_pruned = True
                print("  [!] Trial pruned by Optuna due to poor early performance.")
                raise optuna.exceptions.TrialPruned() 

    finally:
        tf.keras.backend.clear_session()
        gc.collect()
        
        if len(fold_metrics_list) > 0:
            with open(os.path.join(run_dir, "fold_metrics.json"), "w") as f:
                json.dump(fold_metrics_list, f, indent=4)

            avg_metrics = {}
            for metric_name in fold_metrics_list[0].keys():
                if metric_name != "Fold":
                    avg_metrics[f"Avg_{metric_name}"] = round(float(np.mean([m[metric_name] for m in fold_metrics_list])), 3)
                    avg_metrics[f"Std_{metric_name}"] = round(float(np.std([m[metric_name] for m in fold_metrics_list])), 3)
                    
                    for fold_idx in range(K_FOLDS):
                        if fold_idx < len(fold_metrics_list):
                            avg_metrics[f"Fold{fold_idx+1}_{metric_name}"] = fold_metrics_list[fold_idx][metric_name]
                        else:
                            avg_metrics[f"Fold{fold_idx+1}_{metric_name}"] = None

            avg_metrics["Pruned"] = is_pruned 

            print(f"  >>> Trial Finished. Average Val R2: {avg_metrics['Avg_Val_R2']:.3f} (±{avg_metrics['Std_Val_R2']:.3f})\n")

            with open(os.path.join(run_dir, "avg_metrics.json"), "w") as f:
                json.dump(avg_metrics, f, indent=4)

            # Load the best model back strictly for drawing plots
            if os.path.exists(best_model_path) and best_fold_history_dict is not None:
                best_fold_model = tf.keras.models.load_model(best_model_path)
                generate_all_plots(best_fold_model, best_fold_dataset, best_fold_history_dict, plot_dir, eval_batches)

            config_dict = {
                "trial_number": trial.number, "date": readable_date, "total_samples": TOTAL_SAMPLES, 
                "expected_shape": str(EXPECTED_SHAPE), "latent_dim": latent_dim, "learning_rate": lr, 
                "base_filters": base_filters, "kernel_size": kernel_size, "num_layers": num_layers, 
                "stride_size": stride_size, "pooling_type": pooling_type, "activation": activation, 
                "norm_type": norm_type, "dropout_rate": dropout_rate, "batch_size": batch_size, 
                "loss_func": loss_func, "k_folds": K_FOLDS
            }
            
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(config_dict, f, indent=4)

            summary_row = {**config_dict, **avg_metrics}
            summary_df = pd.DataFrame([summary_row])
            csv_path = os.path.join(OUTPUT_DIR, "optuna_summary.csv")
            summary_df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
            
        print(f"EXPERIMENT {trial.number} LOGGED & COMPLETE\n")

    return avg_metrics.get('Avg_Val_R2', -float('inf'))

study = optuna.create_study(direction="maximize", study_name="ECG_Autoencoder_Optimization")
study.optimize(objective, n_trials=N_ITERATIONS)

print("\n" + "="*60)
print("ALL OPTUNA TRIALS COMPLETED!")
print(f"Best Trial Number: {study.best_trial.number}")
print(f"Best R2 Score: {study.best_value:.3f}")
print("Best Hyperparameters:")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")
print("="*60 + "\n") 