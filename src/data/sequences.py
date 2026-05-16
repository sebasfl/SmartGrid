from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DataConfig

# Canonical feature column order: time features followed by value column
DEFAULT_FEATURE_COLS = DataConfig().time_features + [DataConfig().value_col]


def create_sequences(
    df: pd.DataFrame,
    building_id: str,
    lookback: int,
    horizon: int,
    stride: int = 1,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Create sliding window sequences for a single building.

    Args:
        feature_cols: Ordered list of columns to use. Defaults to
            time features + value from DataConfig. The last column
            is used as the forecast target.

    Returns:
        Tuple of (X_sequences, y_forecast) or (None, None) if insufficient data.
            - X_sequences: [num_seqs, lookback, n_features]
            - y_forecast: [num_seqs, horizon]
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    building_df = df[df['building_id'] == building_id].copy()
    building_df = building_df.sort_values('timestamp_local')

    if len(building_df) < lookback + horizon:
        return None, None

    features = building_df[feature_cols].values

    X_sequences: list[np.ndarray] = []
    y_forecast: list[np.ndarray] = []

    for i in range(0, len(features) - lookback - horizon + 1, stride):
        X_sequences.append(features[i:i + lookback])
        y_forecast.append(features[i + lookback:i + lookback + horizon, -1])

    if len(X_sequences) == 0:
        return None, None

    return (
        np.array(X_sequences, dtype=np.float32),
        np.array(y_forecast, dtype=np.float32),
    )
