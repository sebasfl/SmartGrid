from __future__ import annotations

import gc
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from ..config import Config
from ..exceptions import DataQualityError, InsufficientDataError
from .sequences import create_sequences

logger = logging.getLogger(__name__)


def prepare_dataset(
    df: pd.DataFrame,
    config: Config,
    split: str = 'train',
    scaler: StandardScaler | None = None,
    max_buildings: int | None = None,
    building_ids: list[str] | None = None,
    user_specified_n: bool = False,
) -> tuple[tf.data.Dataset, StandardScaler]:
    """Prepare TensorFlow dataset for CNN-LSTM training.

    Args:
        df: Preprocessed dataframe.
        config: Configuration object.
        split: One of 'train', 'val', or 'test'.
        scaler: Pre-fitted scaler (required for val/test).
        max_buildings: Maximum number of buildings to use.
        building_ids: Explicit list of building IDs. Falls back to random split if None.
        user_specified_n: Whether the user explicitly set --n_buildings.

    Returns:
        Tuple of (tf.data.Dataset, fitted StandardScaler).

    Raises:
        InsufficientDataError: If no buildings produce valid sequences.
    """
    logger.info("Preparing %s dataset...", split)

    if building_ids is not None:
        split_buildings = np.array(building_ids)
        logger.info("  Using provided building list: %d buildings", len(split_buildings))
    else:
        all_building_ids = df['building_id'].unique()
        n_buildings = len(all_building_ids)
        np.random.shuffle(all_building_ids)

        train_end = int(n_buildings * config.data.train_ratio)
        val_end = int(n_buildings * (config.data.train_ratio + config.data.val_ratio))

        if split == 'train':
            split_buildings = all_building_ids[:train_end]
        elif split == 'val':
            split_buildings = all_building_ids[train_end:val_end]
        else:
            split_buildings = all_building_ids[val_end:]

        logger.info("  Using random split: %d buildings", len(split_buildings))

    if max_buildings and len(split_buildings) > max_buildings:
        original_count = len(split_buildings)
        split_buildings = split_buildings[:max_buildings]
        if user_specified_n:
            logger.info("  Using first %d buildings (user-specified)", max_buildings)
        else:
            logger.info("  Limited to first %d of %d buildings (memory optimization)", max_buildings, original_count)

    logger.info("  %s buildings: %d", split.capitalize(), len(split_buildings))

    if split == 'train':
        n_sample = getattr(config.data, 'scaler_sample_buildings', 10)
        logger.info("  Fitting scaler on sample (%d buildings)...", n_sample)
        sample_buildings = split_buildings[:min(n_sample, len(split_buildings))]
        scaler = StandardScaler()

        for i, building_id in enumerate(sample_buildings):
            X, _ = create_sequences(
                df, building_id,
                lookback=config.data.lookback_window,
                horizon=config.data.forecast_horizon,
                stride=config.data.stride * 10,
            )
            if X is not None and len(X) > 0:
                X_sample = X[:min(100, len(X))]
                scaler.partial_fit(X_sample.reshape(-1, X_sample.shape[-1]))

        import sklearn
        scaler_path = Path(config.data.model_dir) / 'scaler.pkl'
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'scaler': scaler, 'sklearn_version': sklearn.__version__}, scaler_path)
        logger.info("  Saved scaler to %s (sklearn %s)", scaler_path, sklearn.__version__)

    def data_generator() -> tf.data.Dataset:
        total_sequences = 0
        skipped_invalid = 0

        for i, building_id in enumerate(split_buildings):
            X, y_f = create_sequences(
                df, building_id,
                lookback=config.data.lookback_window,
                horizon=config.data.forecast_horizon,
                stride=config.data.stride,
            )

            if X is not None and y_f is not None:
                if scaler is not None:
                    X_reshaped = X.reshape(-1, X.shape[-1])
                    X = scaler.transform(X_reshaped).reshape(X.shape)

                for j in range(len(X)):
                    if np.isnan(X[j]).any() or np.isnan(y_f[j]).any():
                        skipped_invalid += 1
                        continue
                    if np.isinf(X[j]).any() or np.isinf(y_f[j]).any():
                        skipped_invalid += 1
                        continue

                    yield X[j], y_f[j]
                    total_sequences += 1

                del X, y_f

            if (i + 1) % 50 == 0:
                gc.collect()

        if total_sequences == 0:
            min_records = config.data.lookback_window + config.data.forecast_horizon
            raise InsufficientDataError(
                f"No valid sequences produced for '{split}' split from "
                f"{len(split_buildings)} buildings. "
                f"Each building needs >= {min_records} records "
                f"(lookback={config.data.lookback_window} + "
                f"horizon={config.data.forecast_horizon}), "
                f"stride={config.data.stride}."
            )

        if skipped_invalid > 0:
            invalid_pct = 100 * skipped_invalid / (total_sequences + skipped_invalid)
            logger.warning(
                "  Skipped %d invalid sequences (%.1f%%) containing NaN/Inf",
                skipped_invalid, invalid_pct,
            )
            if invalid_pct > 50:
                raise DataQualityError(
                    f"{invalid_pct:.0f}% of sequences contain NaN/Inf values. "
                    f"Run data quality analysis before training."
                )

    output_signature = (
        tf.TensorSpec(shape=(config.data.lookback_window, config.data.input_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(config.data.forecast_horizon,), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(data_generator, output_signature=output_signature)

    if split == 'train':
        dataset = dataset.shuffle(buffer_size=2000)

    dataset = dataset.batch(config.training.batch_size).prefetch(2)

    logger.info("  %s dataset created (lazy loading enabled)", split.capitalize())
    return dataset, scaler
