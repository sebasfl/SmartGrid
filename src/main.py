from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .config import Config
from .data.dataset import prepare_dataset
from .data.loading import filter_by_building_ids, load_building_split, load_parquet
from .exceptions import DataError
from .models.cnn_lstm import build_cnn_lstm_model
from .training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoardLogger,
    TimerCallback,
)
from .training.trainer import CNNLSTMTrainer
from .utils.gpu import check_gpu_availability, configure_gpu
from .utils.reproducibility import set_random_seed

logger = logging.getLogger(__name__)


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Remove NaN and infinite values, logging what was discarded."""
    original_len = len(df)

    nan_count = df.isnull().sum().sum()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = int(np.isinf(df[numeric_cols]).any(axis=1).sum())

    if nan_count == 0 and inf_count == 0:
        return df

    mask = ~(df.isnull().any(axis=1) | np.isinf(df[numeric_cols]).any(axis=1))
    df_clean = df[mask]
    removed = original_len - len(df_clean)
    removed_pct = 100 * removed / original_len

    logger.warning(
        "Removed %d rows (%.1f%%): %d NaN, %d Inf",
        removed, removed_pct, nan_count, inf_count,
    )

    if removed_pct > 20:
        raise DataError(
            f"Data quality issue: {removed_pct:.0f}% of rows contain NaN/Inf. "
            f"Run preprocessing and data quality analysis first."
        )

    return df_clean


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    configure_gpu()
    check_gpu_availability()

    if args.config:
        config = Config.from_json(args.config)
        logger.info("Loaded config from %s", args.config)
    else:
        config = Config()
        logger.info("Using default configuration")

    if args.parquet:
        config.data.parquet_path = args.parquet
    if args.model_dir:
        config.data.model_dir = args.model_dir
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size

    config.validate()
    logger.info("\n%s", config)

    set_random_seed(config.training.random_seed)

    df = load_parquet(config.data.parquet_path)
    df = _clean_dataframe(df)

    train_building_ids: list[str] | None = None
    val_building_ids: list[str] | None = None

    if args.building_split:
        logger.info("Loading building split from %s...", args.building_split)
        building_split = load_building_split(args.building_split)

        train_building_ids = building_split.get('train', [])
        val_building_ids = building_split.get('validation', [])
        test_building_ids = building_split.get('test', [])

        logger.info("  Train: %d | Val: %d | Test: %d (reserved)",
                     len(train_building_ids), len(val_building_ids), len(test_building_ids))

        training_quality_buildings = train_building_ids + val_building_ids
        df = filter_by_building_ids(df, training_quality_buildings)

    if args.n_buildings is not None:
        max_train_buildings: int | None = args.n_buildings
        max_val_buildings: int | None = max(int(args.n_buildings * 0.25), 1)
        user_specified_n = True
    elif args.use_full_dataset:
        max_train_buildings = None
        max_val_buildings = None
        user_specified_n = False
    else:
        max_train_buildings = 150
        max_val_buildings = 40
        user_specified_n = False

    train_dataset, scaler = prepare_dataset(
        df, config, split='train',
        max_buildings=max_train_buildings,
        building_ids=train_building_ids,
        user_specified_n=user_specified_n,
    )

    gc.collect()

    val_dataset, _ = prepare_dataset(
        df, config, split='val',
        scaler=scaler,
        max_buildings=max_val_buildings,
        building_ids=val_building_ids,
        user_specified_n=user_specified_n,
    )

    logger.info("Building CNN-LSTM model...")

    model = build_cnn_lstm_model(
        input_shape=(config.data.lookback_window, config.data.input_dim),
        forecast_horizon=config.data.forecast_horizon,
        cnn_config=config.cnn,
        lstm_config=config.lstm,
        forecast_config=config.forecast_head,
    )

    logger.info("  Total parameters: %s", f"{model.count_params():,}")

    loss_fns = {
        'mse': tf.keras.losses.MeanSquaredError,
        'mae': tf.keras.losses.MeanAbsoluteError,
        'huber': lambda: tf.keras.losses.Huber(delta=config.loss.huber_delta),
    }
    loss_fn = loss_fns.get(config.loss.forecast_loss_type, tf.keras.losses.MeanSquaredError)()

    trainer = CNNLSTMTrainer(model, loss_fn, config.optimizer, config.training)

    model_dir = Path(config.data.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor=config.training.early_stopping_monitor,
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta,
        ),
        ModelCheckpoint(
            filepath=str(model_dir / 'best_model_epoch_{epoch}.h5'),
            monitor='val_loss', save_best_only=True,
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=config.optimizer.min_lr),
        TensorBoardLogger(log_dir=config.training.tensorboard_dir),
        TimerCallback(),
    ]

    trainer.fit(train_dataset, val_dataset, callbacks=callbacks)

    final_model_path = model_dir / 'cnn_lstm_final.h5'
    model.save_weights(str(final_model_path))
    logger.info("Final model saved to %s", final_model_path)

    trainer.save_history(str(model_dir / 'training_history.json'))

    config_path = model_dir / 'config.json'
    config.to_json(str(config_path))
    logger.info("Configuration saved to %s", config_path)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CNN-LSTM for energy forecasting')
    parser.add_argument('--config', type=str, default=None, help='Path to config JSON file')
    parser.add_argument('--parquet', type=str, default=None, help='Path to preprocessed parquet file')
    parser.add_argument('--model_dir', type=str, default=None, help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--use_full_dataset', action='store_true', help='Use all buildings')
    parser.add_argument('--n_buildings', type=int, default=None, help='Number of buildings to use')
    parser.add_argument('--building_split', type=str, default=None, help='Path to building_split.json')

    args = parser.parse_args()
    main(args)
