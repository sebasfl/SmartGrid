import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")

from src.config import Config
from src.data.sequences import create_sequences
from src.models.cnn_lstm import build_cnn_lstm_model
from src.training.trainer import CNNLSTMTrainer
from src.evaluation.metrics import ForecastMetrics

EXPECTED_METRIC_KEYS = {'rmse', 'mae', 'mape', 'smape', 'r2', 'nrmse'}


def _make_synthetic_df(n_buildings=3, n_timestamps=500, freq='3h', seed=42):
    np.random.seed(seed)
    rows = []
    for i in range(n_buildings):
        bid = f'building_{i}'
        timestamps = pd.date_range('2020-01-01', periods=n_timestamps, freq=freq)
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


def _small_config():
    config = Config()
    config.data.lookback_window = 24
    config.data.forecast_horizon = 12
    config.data.stride = 4
    config.training.batch_size = 2
    config.training.epochs = 1
    config.training.use_mixed_precision = False
    config.training.log_freq = 10
    config.cnn.filters = [16, 32]
    config.cnn.kernel_sizes = [3, 3]
    config.lstm.units = [16]
    config.forecast_head.hidden_dims = [16]
    return config


class TestEndToEndPipeline:

    def test_full_pipeline_synthetic_data(self):
        config = _small_config()
        df = _make_synthetic_df(n_buildings=3, n_timestamps=500)
        building_id = 'building_0'

        X, y = create_sequences(
            df, building_id,
            lookback=config.data.lookback_window,
            horizon=config.data.forecast_horizon,
            stride=config.data.stride,
        )
        assert X is not None and y is not None

        input_shape = (config.data.lookback_window, 8)
        model = build_cnn_lstm_model(
            input_shape=input_shape,
            forecast_horizon=config.data.forecast_horizon,
            cnn_config=config.cnn,
            lstm_config=config.lstm,
            forecast_config=config.forecast_head,
        )

        split = int(len(X) * 0.8)
        train_ds = tf.data.Dataset.from_tensor_slices((X[:split], y[:split])).batch(config.training.batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((X[split:], y[split:])).batch(config.training.batch_size)

        trainer = CNNLSTMTrainer(
            model=model,
            loss_fn=tf.keras.losses.MeanSquaredError(),
            optimizer_config=config.optimizer,
            training_config=config.training,
        )
        history = trainer.fit(train_ds, val_ds)

        assert len(history['train_loss']) == 1
        assert np.isfinite(history['train_loss'][0])

        batch_x = X[:4]
        preds = model(batch_x, training=False).numpy()
        assert preds.shape == (4, config.data.forecast_horizon)

        metrics = ForecastMetrics.compute_all(y[:4], preds)
        assert set(metrics.keys()) == EXPECTED_METRIC_KEYS
        assert all(np.isfinite(v) for v in metrics.values())

    def test_model_output_shape_matches_horizon(self, default_config):
        horizon = default_config.data.forecast_horizon
        model = build_cnn_lstm_model(
            input_shape=(default_config.data.lookback_window, 8),
            forecast_horizon=horizon,
            cnn_config=default_config.cnn,
            lstm_config=default_config.lstm,
            forecast_config=default_config.forecast_head,
        )
        x = tf.random.normal((3, default_config.data.lookback_window, 8))
        output = model(x, training=False)
        assert output.shape == (3, horizon)

    def test_sequences_feed_into_model(self, sample_building_df, default_config):
        X, y = create_sequences(
            sample_building_df, 'building_A',
            lookback=default_config.data.lookback_window,
            horizon=default_config.data.forecast_horizon,
            stride=default_config.data.stride,
        )
        assert X is not None

        model = build_cnn_lstm_model(
            input_shape=(default_config.data.lookback_window, 8),
            forecast_horizon=default_config.data.forecast_horizon,
            cnn_config=default_config.cnn,
            lstm_config=default_config.lstm,
            forecast_config=default_config.forecast_head,
        )
        output = model(X, training=False)
        assert output.shape[0] == X.shape[0]
        assert output.shape[1] == default_config.data.forecast_horizon
