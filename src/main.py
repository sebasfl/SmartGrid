# src/main.py
# Main training script for MTL Transformer
import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import json
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
from .models.transformer import TransformerEncoder
from .models.mtl_heads import MTLModel, AnomalyDetectionHead, ForecastingHead
from .models.losses import MTLLoss
from .training.trainer import MTLTrainer
from .training.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoardLogger, TimerCallback
)
from .evaluation.metrics import MTLMetrics


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
        Tuple of (X_sequences, y_anomaly, y_forecast)
    """
    # Filter building data
    building_df = df[df['building_id'] == building_id].copy()
    building_df = building_df.sort_values('timestamp_local')

    if len(building_df) < lookback + horizon:
        return None, None, None

    # Feature columns (exclude metadata)
    feature_cols = ['hour', 'day_of_week', 'month', 'is_weekend',
                    'is_working_hours', 'quarter', 'day_of_year', 'value']

    # Extract features
    features = building_df[feature_cols].values

    # Create sequences
    X_sequences = []
    y_anomaly = []
    y_forecast = []

    for i in range(0, len(features) - lookback - horizon + 1, stride):
        # Input sequence
        X_seq = features[i:i + lookback]

        # Anomaly label: check if there are anomalies in the lookback window
        # Using IQR method for labeling
        values = X_seq[:, -1]  # Last column is 'value'
        Q1, Q3 = np.percentile(values, [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        has_anomaly = np.any((values < lower_bound) | (values > upper_bound))
        y_anom = 1.0 if has_anomaly else 0.0

        # Forecast target: next 'horizon' values
        y_fore = features[i + lookback:i + lookback + horizon, -1]

        X_sequences.append(X_seq)
        y_anomaly.append(y_anom)
        y_forecast.append(y_fore)

    if len(X_sequences) == 0:
        return None, None, None

    X = np.array(X_sequences, dtype=np.float32)
    y_a = np.array(y_anomaly, dtype=np.float32).reshape(-1, 1)
    y_f = np.array(y_forecast, dtype=np.float32)

    return X, y_a, y_f


def prepare_mtl_dataset(df: pd.DataFrame, config: Config,
                       split: str = 'train', scaler=None, max_buildings: int = None) -> tf.data.Dataset:
    """Prepare TensorFlow dataset for MTL training with memory-efficient batch processing.

    Args:
        df: Preprocessed dataframe
        config: Configuration object
        split: 'train', 'val', or 'test'
        scaler: StandardScaler for normalization (required for val/test)
        max_buildings: Maximum number of buildings to use (for memory efficiency)

    Returns:
        TensorFlow dataset
    """
    print(f"\n📦 Preparing {split} dataset...")

    building_ids = df['building_id'].unique()
    n_buildings = len(building_ids)

    # Split buildings by train/val/test
    np.random.shuffle(building_ids)
    train_end = int(n_buildings * config.data.train_ratio)
    val_end = int(n_buildings * (config.data.train_ratio + config.data.val_ratio))

    if split == 'train':
        split_buildings = building_ids[:train_end]
    elif split == 'val':
        split_buildings = building_ids[train_end:val_end]
    else:  # test
        split_buildings = building_ids[val_end:]

    # Limit buildings for memory efficiency
    if max_buildings and len(split_buildings) > max_buildings:
        split_buildings = split_buildings[:max_buildings]
        print(f"   🔍 Limited to {max_buildings} buildings (memory optimization)")

    print(f"   {split.capitalize()} buildings: {len(split_buildings)}")

    # Fit scaler on a small sample if training (memory-efficient)
    if split == 'train':
        print("   Fitting scaler on sample (10 buildings)...")
        sample_buildings = split_buildings[:min(10, len(split_buildings))]
        scaler = StandardScaler()

        # Fit incrementally to avoid memory issues
        for i, building_id in enumerate(sample_buildings):
            X, _, _ = create_sequences(
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
        anomaly_count = 0
        skipped_nan = 0

        for i, building_id in enumerate(split_buildings):
            X, y_a, y_f = create_sequences(
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

                    yield X[j], y_a[j], y_f[j]
                    total_sequences += 1
                    anomaly_count += y_a[j][0]

                # Free memory after processing building
                del X, y_a, y_f, X_reshaped, X_normalized

            if (i + 1) % 50 == 0:
                print(f"   Processed {i + 1}/{len(split_buildings)} buildings... (skipped {skipped_nan} NaN sequences)")
                gc.collect()  # Force garbage collection every 50 buildings

    # Define output signature
    input_dim = len(config.data.time_features) + 1
    output_signature = (
        tf.TensorSpec(shape=(config.data.lookback_window, input_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(1,), dtype=tf.float32),
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


def build_mtl_model(config: Config) -> MTLModel:
    """Build complete MTL model.

    Args:
        config: Configuration object

    Returns:
        MTL model
    """
    print("\n🏗️  Building MTL model...")

    # Input dimension: number of features
    input_dim = len(config.data.time_features) + 1  # time features + value

    # Transformer encoder
    encoder = TransformerEncoder(
        num_layers=config.transformer.num_layers,
        d_model=config.transformer.d_model,
        num_heads=config.transformer.num_heads,
        ff_dim=config.transformer.ff_dim,
        input_dim=input_dim,
        dropout=config.transformer.dropout,
        activation=config.transformer.activation,
        use_positional_encoding=config.transformer.use_positional_encoding,
        positional_encoding_type=config.transformer.positional_encoding_type,
        max_seq_length=config.transformer.max_seq_length
    )

    # Anomaly detection head
    anomaly_head = AnomalyDetectionHead(
        hidden_dims=config.mtl_heads.anomaly_hidden_dims,
        dropout=config.mtl_heads.anomaly_dropout,
        activation=config.mtl_heads.anomaly_activation
    )

    # Forecasting head
    forecast_head = ForecastingHead(
        forecast_horizon=config.data.forecast_horizon,
        hidden_dims=config.mtl_heads.forecast_hidden_dims,
        dropout=config.mtl_heads.forecast_dropout,
        activation=config.mtl_heads.forecast_activation,
        use_decoder=config.mtl_heads.use_decoder
    )

    # Complete MTL model
    model = MTLModel(encoder, anomaly_head, forecast_head)

    # Build model with dummy input
    dummy_input = tf.random.normal((1, config.data.lookback_window, input_dim))
    _ = model(dummy_input, training=False)

    # Print model summary
    total_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
    print(f"   Total parameters: {total_params:,}")

    return model


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

    # Prepare datasets (memory-efficient batch processing)
    # Set to None to use ALL buildings (requires 6GB+ RAM)
    # Set to a number to limit (200-500 recommended for 2-4GB RAM)
    max_train_buildings = None if args.use_full_dataset else 250
    max_val_buildings = None if args.use_full_dataset else 50

    if args.use_full_dataset:
        print(f"\n⚙️  Using FULL dataset (all {df['building_id'].nunique()} buildings)")
    else:
        print(f"\n⚙️  Memory optimization: Using max {max_train_buildings} train buildings, {max_val_buildings} val buildings")

    train_dataset, scaler = prepare_mtl_dataset(df, config, split='train', max_buildings=max_train_buildings)

    # Force garbage collection before validation dataset
    import gc
    gc.collect()

    val_dataset, _ = prepare_mtl_dataset(df, config, split='val', scaler=scaler, max_buildings=max_val_buildings)

    # Build model
    model = build_mtl_model(config)

    # Setup loss function
    loss_fn = MTLLoss(
        alpha=config.loss.alpha_anomaly,
        beta=config.loss.beta_forecast,
        anomaly_loss_type=config.loss.anomaly_loss_type,
        forecast_loss_type=config.loss.forecast_loss_type,
        use_uncertainty_weighting=config.loss.use_uncertainty_weighting
    )

    # Create trainer
    trainer = MTLTrainer(model, loss_fn, config.optimizer, config.training)

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
    final_model_path = model_dir / 'mtl_transformer_final.h5'
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
    parser = argparse.ArgumentParser(description='Train MTL Transformer for energy forecasting')

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

    args = parser.parse_args()
    main(args)
