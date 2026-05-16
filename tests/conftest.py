import numpy as np
import pandas as pd
import pytest

from src.config import Config


@pytest.fixture
def sample_building_df():
    """Small DataFrame with 2 buildings, 200 timestamps each."""
    np.random.seed(42)
    rows = []
    for bid in ['building_A', 'building_B']:
        timestamps = pd.date_range('2020-01-01', periods=200, freq='3h')
        for ts in timestamps:
            rows.append({
                'timestamp_local': ts,
                'building_id': bid,
                'meter': 'electricity',
                'value': np.random.uniform(10, 100),
                'hour': ts.hour,
                'day_of_week': ts.dayofweek,
                'month': ts.month,
                'is_weekend': int(ts.dayofweek >= 5),
                'is_working_hours': int(8 <= ts.hour <= 18 and ts.dayofweek < 5),
                'quarter': ts.quarter,
                'day_of_year': ts.dayofyear,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def default_config():
    """Config with reduced sizes for fast testing."""
    config = Config()
    config.data.lookback_window = 24
    config.data.forecast_horizon = 12
    config.data.stride = 4
    config.training.batch_size = 2
    config.training.epochs = 1
    config.cnn.filters = [16, 32]
    config.cnn.kernel_sizes = [3, 3]
    config.lstm.units = [16]
    config.forecast_head.hidden_dims = [16]
    return config
