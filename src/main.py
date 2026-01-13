# src/main.py
# Main training script for Hybrid CNN-LSTM model
import os

# ============================================================================
# CRITICAL: Disable cuDNN for RNNs (cuDNN 9.0+ compatibility)
# MUST BE SET BEFORE IMPORTING TENSORFLOW
# ============================================================================
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import List
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GPU CONFIGURATION - MUST BE DONE BEFORE ANY TF OPERATIONS
# ============================================================================
# Configure GPU memory growth and visibility BEFORE any TensorFlow operations
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Enable memory growth for all GPUs (prevents TF from allocating all GPU memory at once)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Set visible devices to first GPU (if multiple GPUs available)
        tf.config.set_visible_devices(gpus[0], 'GPU')

        print(f"✅ GPU Configuration: {len(gpus)} GPU(s) found, memory growth enabled")
    else:
        print("⚠️  No GPU detected - will run on CPU")
except RuntimeError as e:
    print(f"⚠️  GPU configuration error: {e}")
# ============================================================================

# Local imports
from .config import Config
from .models.cnn_lstm import build_cnn_lstm_model
from .training.trainer import CNNLSTMTrainer
from .training.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoardLogger, TimerCallback
)


def check_gpu_availability():
    """Check and print GPU availability status."""
    print("\n🔍 GPU Availability Check:")
    print(f"   TensorFlow version: {tf.__version__}")

    # Check GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   ✅ {len(gpus)} GPU(s) detected:")
        for i, gpu in enumerate(gpus):
            print(f"      GPU {i}: {gpu.name}")
    else:
        print("   ❌ No GPU detected - training will use CPU")

    # Check CUDA availability
    print(f"   CUDA built: {tf.test.is_built_with_cuda()}")
    gpu_device = tf.test.gpu_device_name()
    if gpu_device:
        print(f"   GPU device name: {gpu_device}")
    else:
        print(f"   GPU device name: None (CPU mode)")

    # Check visible devices
    visible_devices = tf.config.get_visible_devices('GPU')
    print(f"   Visible GPU devices: {len(visible_devices)}")

    return len(gpus) > 0


def set_random_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_sequences(df: pd.DataFrame, building_id: str, lookback: int,
                     horizon: int, stride: int = 1) -> tuple:
    """Create sliding window sequences for a single building.

    Args:
        df: DataFrame with preprocessed data
        building_id: Building ID to process
        lookback: Number of historical timesteps
        horizon: Number of future timesteps to predict
        stride: Stride for sliding window

    Returns:
        Tuple of (X_sequences, y_forecast)
            - X_sequences: [num_seqs, lookback, 8] - input sequences
            - y_forecast: [num_seqs, horizon] - future values to predict
    """
    # Filter building data
    building_df = df[df['building_id'] == building_id].copy()
    building_df = building_df.sort_values('timestamp_local')

    if len(building_df) < lookback + horizon:
        return None, None

    # Feature columns (exclude metadata)
    feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend',
                    'is_working_hours', 'quarter', 'day_of_year', 'value']

    # Extract features
    features = building_df[feature_cols].values

    # Create sequences
    X_sequences = []
    y_forecast = []

    for i in range(0, len(features) - lookback - horizon + 1, stride):
        # Input sequence
        X_seq = features[i:i + lookback]

        # Forecast target: next 'horizon' values
        y_fore = features[i + lookback:i + lookback + horizon, -1]

        X_sequences.append(X_seq)
        y_forecast.append(y_fore)

    if len(X_sequences) == 0:
        return None, None

    X = np.array(X_sequences, dtype=np.float32)
    y_f = np.array(y_forecast, dtype=np.float32)

    return X, y_f


def prepare_dataset(df: pd.DataFrame, config: Config,
                   split: str = 'train', scaler=None, max_buildings: int = None,
                   building_ids: List[str] = None, user_specified_n: bool = False) -> tf.data.Dataset:
    """Prepare TensorFlow dataset for CNN-LSTM training with memory-efficient batch processing.

    Args:
        df: Preprocessed dataframe
        config: Configuration object
        split: 'train', 'val', or 'test' (used for logging only if building_ids provided)
        scaler: StandardScaler for normalization (required for val/test)
        max_buildings: Maximum number of buildings to use (for memory efficiency)
        building_ids: Optional list of building IDs to use. If None, uses random split.
        user_specified_n: True if user specified --n_buildings (for better logging)

    Returns:
        TensorFlow dataset and scaler
    """
    print(f"\n📦 Preparing {split} dataset...")

    # Use provided building IDs or fall back to random split
    if building_ids is not None:
        split_buildings = np.array(building_ids)
        print(f"   Using provided building list: {len(split_buildings)} buildings")
    else:
        # Legacy random split (if no building_split.json provided)
        all_building_ids = df['building_id'].unique()
        n_buildings = len(all_building_ids)

        np.random.shuffle(all_building_ids)
        train_end = int(n_buildings * config.data.train_ratio)
        val_end = int(n_buildings * (config.data.train_ratio + config.data.val_ratio))

        if split == 'train':
            split_buildings = all_building_ids[:train_end]
        elif split == 'val':
            split_buildings = all_building_ids[train_end:val_end]
        else:  # test
            split_buildings = all_building_ids[val_end:]

        print(f"   Using random split: {len(split_buildings)} buildings")

    # Limit buildings for memory efficiency or user specification
    if max_buildings and len(split_buildings) > max_buildings:
        original_count = len(split_buildings)
        split_buildings = split_buildings[:max_buildings]

        if user_specified_n:
            print(f"   🎯 Using first {max_buildings} buildings (user-specified)")
        else:
            print(f"   🔍 Limited to first {max_buildings} of {original_count} buildings (memory optimization)")

    print(f"   {split.capitalize()} buildings: {len(split_buildings)}")

    # Fit scaler on a small sample if training (memory-efficient)
    if split == 'train':
        print("   Fitting scaler on sample (10 buildings)...")
        sample_buildings = split_buildings[:min(10, len(split_buildings))]
        scaler = StandardScaler()

        # Fit incrementally to avoid memory issues
        for i, building_id in enumerate(sample_buildings):
            X, _ = create_sequences(
                df, building_id,
                lookback=config.data.lookback_window,
                horizon=config.data.forecast_horizon,
                stride=config.data.stride * 10  # Use larger stride for sampling
            )
            if X is not None and len(X) > 0:
                # Take only first 100 sequences max per building
                X_sample = X[:min(100, len(X))]
                X_reshaped = X_sample.reshape(-1, X_sample.shape[-1])
                scaler.partial_fit(X_reshaped)
                print(f"      Fitted building {i+1}/{len(sample_buildings)}")

        # Save scaler
        scaler_path = Path(config.data.model_dir) / 'scaler.pkl'
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"   ✅ Saved scaler to {scaler_path}")

    # Generator function for lazy loading
    def data_generator():
        """Generate sequences on-demand to avoid loading all data in memory."""
        import gc
        total_sequences = 0
        skipped_nan = 0

        for i, building_id in enumerate(split_buildings):
            X, y_f = create_sequences(
                df, building_id,
                lookback=config.data.lookback_window,
                horizon=config.data.forecast_horizon,
                stride=config.data.stride
            )

            if X is not None:
                # Normalize using scaler
                if scaler is not None:
                    X_reshaped = X.reshape(-1, X.shape[-1])
                    X_normalized = scaler.transform(X_reshaped)
                    X = X_normalized.reshape(X.shape)

                # Yield individual sequences (skip NaN sequences)
                for j in range(len(X)):
                    # Check for NaN in sequence
                    if np.isnan(X[j]).any() or np.isnan(y_f[j]).any():
                        skipped_nan += 1
                        continue

                    # Check for infinite values
                    if np.isinf(X[j]).any() or np.isinf(y_f[j]).any():
                        skipped_nan += 1
                        continue

                    yield X[j], y_f[j]
                    total_sequences += 1

                # Free memory after processing building
                del X, y_f, X_reshaped, X_normalized

            if (i + 1) % 50 == 0:
                print(f"   Processed {i + 1}/{len(split_buildings)} buildings... (skipped {skipped_nan} NaN sequences)")
                gc.collect()  # Force garbage collection every 50 buildings

    # Define output signature
    input_dim = len(config.data.time_features) + 1
    output_signature = (
        tf.TensorSpec(shape=(config.data.lookback_window, input_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(config.data.forecast_horizon,), dtype=tf.float32)
    )

    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=output_signature
    )

    # Shuffle and batch (reduced buffer for memory efficiency)
    if split == 'train':
        dataset = dataset.shuffle(buffer_size=2000)  # Reduced from 10000 to 2000

    dataset = dataset.batch(config.training.batch_size)
    dataset = dataset.prefetch(2)  # Reduced from AUTOTUNE to 2

    print(f"   ✅ {split.capitalize()} dataset created (lazy loading enabled)")

    return dataset, scaler


def main(args):
    """Main training function."""

    # Check GPU availability
    gpu_available = check_gpu_availability()

    # Load configuration
    if args.config:
        config = Config.from_json(args.config)
        print(f"📋 Loaded config from {args.config}")
    else:
        config = Config()
        print("📋 Using default configuration")

    # Override config with command-line args
    if args.parquet:
        config.data.parquet_path = args.parquet
    if args.model_dir:
        config.data.model_dir = args.model_dir
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size

    print(config)

    # Set random seed
    set_random_seed(config.training.random_seed)

    # Load preprocessed data
    print(f"\n📥 Loading data from {config.data.parquet_path}...")
    df = pd.read_parquet(config.data.parquet_path)
    print(f"   Loaded {len(df):,} records for {df['building_id'].nunique()} buildings")

    # Check for and handle NaN values
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(f"   ⚠️  Found {nan_count} NaN values in data")
        print(f"   Dropping rows with NaN...")
        df = df.dropna()
        print(f"   After cleaning: {len(df):,} records for {df['building_id'].nunique()} buildings")

    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(df[numeric_cols]).any(axis=1)
    if inf_mask.sum() > 0:
        print(f"   ⚠️  Found {inf_mask.sum()} rows with infinite values")
        print(f"   Dropping rows with infinite values...")
        df = df[~inf_mask]
        print(f"   After cleaning: {len(df):,} records")

    # Load building split if provided (from data quality analysis)
    train_building_ids = None
    val_building_ids = None
    test_building_ids = None

    if args.building_split:
        print(f"\n📂 Loading building split from {args.building_split}...")
        with open(args.building_split, 'r') as f:
            building_split = json.load(f)

        train_building_ids = building_split.get('train', [])
        val_building_ids = building_split.get('validation', [])
        test_building_ids = building_split.get('test', [])

        print(f"   Training buildings:   {len(train_building_ids)}")
        print(f"   Validation buildings: {len(val_building_ids)}")
        print(f"   Test buildings:       {len(test_building_ids)} (reserved for final evaluation)")

        # Filter dataframe to only include high-quality buildings (train + val only)
        # Test buildings are kept separate for final evaluation
        training_quality_buildings = train_building_ids + val_building_ids
        df = df[df['building_id'].isin(training_quality_buildings)]
        print(f"   Filtered to high-quality buildings (train+val): {len(df):,} records")

    # Prepare datasets (memory-efficient batch processing)
    # Determine max buildings based on user input
    user_specified_n = args.n_buildings is not None

    if args.n_buildings is not None:
        # User specified exact number of buildings
        max_train_buildings = args.n_buildings
        max_val_buildings = max(int(args.n_buildings * 0.25), 1)  # 25% of train buildings, min 1
        print(f"\n⚙️  User Selection: First {max_train_buildings} train buildings and {max_val_buildings} validation buildings")
    elif args.use_full_dataset:
        # Use ALL buildings (requires 6GB+ RAM)
        max_train_buildings = None
        max_val_buildings = None
        print(f"\n⚙️  Using FULL dataset (all available buildings)")
    else:
        # Default: memory-optimized limits
        max_train_buildings = 150
        max_val_buildings = 40
        print(f"\n⚙️  Memory optimization: Limited to {max_train_buildings} train / {max_val_buildings} val buildings")

    train_dataset, scaler = prepare_dataset(
        df, config, split='train',
        max_buildings=max_train_buildings,
        building_ids=train_building_ids,
        user_specified_n=user_specified_n
    )

    # Force garbage collection before validation dataset
    import gc
    gc.collect()

    val_dataset, _ = prepare_dataset(
        df, config, split='val',
        scaler=scaler,
        max_buildings=max_val_buildings,
        building_ids=val_building_ids,
        user_specified_n=user_specified_n
    )

    # Build model
    print("\n🏗️  Building CNN-LSTM model...")
    input_dim = len(config.data.time_features) + 1

    model = build_cnn_lstm_model(
        input_shape=(config.data.lookback_window, input_dim),
        forecast_horizon=config.data.forecast_horizon,
        cnn_config={
            'filters': config.cnn.filters,
            'kernel_sizes': config.cnn.kernel_sizes,
            'dropout': config.cnn.dropout,
            'activation': config.cnn.activation,
            'use_batch_norm': config.cnn.use_batch_norm
        },
        lstm_config={
            'units': config.lstm.units,
            'dropout': config.lstm.dropout,
            'recurrent_dropout': config.lstm.recurrent_dropout,
            'use_bidirectional': config.lstm.use_bidirectional
        },
        forecast_config={
            'hidden_dims': config.forecast_head.hidden_dims,
            'dropout': config.forecast_head.dropout,
            'activation': config.forecast_head.activation
        },
        learning_rate=config.optimizer.learning_rate,
        loss=config.loss.forecast_loss_type
    )

    print(f"   Total parameters: {model.count_params():,}")

    # Setup loss function
    if config.loss.forecast_loss_type == 'mse':
        loss_fn = tf.keras.losses.MeanSquaredError()
    elif config.loss.forecast_loss_type == 'mae':
        loss_fn = tf.keras.losses.MeanAbsoluteError()
    elif config.loss.forecast_loss_type == 'huber':
        loss_fn = tf.keras.losses.Huber(delta=config.loss.huber_delta)
    else:
        loss_fn = tf.keras.losses.MeanSquaredError()

    # Create trainer
    trainer = CNNLSTMTrainer(model, loss_fn, config.optimizer, config.training)

    # Setup callbacks
    model_dir = Path(config.data.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor=config.training.early_stopping_monitor,
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta
        ),
        ModelCheckpoint(
            filepath=str(model_dir / 'best_model_epoch_{epoch}.h5'),
            monitor='val_loss',
            save_best_only=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=config.optimizer.min_lr
        ),
        TensorBoardLogger(
            log_dir=config.training.tensorboard_dir
        ),
        TimerCallback()
    ]

    # Train model
    history = trainer.fit(train_dataset, val_dataset, callbacks=callbacks)

    # Save final model
    final_model_path = model_dir / 'cnn_lstm_final.h5'
    model.save_weights(str(final_model_path))
    print(f"\n💾 Final model saved to {final_model_path}")

    # Save training history
    history_path = model_dir / 'training_history.json'
    trainer.save_history(str(history_path))

    # Save configuration
    config_path = model_dir / 'config.json'
    config.to_json(str(config_path))
    print(f"💾 Configuration saved to {config_path}")

    print("\n🎉 Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CNN-LSTM for energy forecasting')

    parser.add_argument('--config', type=str, default=None,
                        help='Path to config JSON file')
    parser.add_argument('--parquet', type=str, default=None,
                        help='Path to preprocessed parquet file')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--use_full_dataset', action='store_true',
                        help='Use all buildings (requires 6GB+ RAM)')
    parser.add_argument('--n_buildings', type=int, default=None,
                        help='Number of buildings to use (e.g., 10 for first 10 buildings). Overrides use_full_dataset.')
    parser.add_argument('--building_split', type=str, default=None,
                        help='Path to building_split.json (from data quality analysis)')

    args = parser.parse_args()
    main(args)
