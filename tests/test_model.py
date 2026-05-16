import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.models.cnn_lstm import (
    CNNFeatureExtractor,
    ForecastingHead,
    HybridCNNLSTM,
    LSTMTemporalEncoder,
    build_cnn_lstm_model,
)


class TestBuildModel:
    def test_returns_built_model(self):
        model = build_cnn_lstm_model(
            input_shape=(24, 8),
            forecast_horizon=12,
        )
        assert model is not None
        assert model.built

    def test_output_shape(self):
        model = build_cnn_lstm_model(
            input_shape=(24, 8),
            forecast_horizon=12,
        )
        dummy_input = tf.random.normal((2, 24, 8))
        output = model(dummy_input, training=False)
        assert output.shape == (2, 12)

    def test_forward_pass_no_error(self):
        model = build_cnn_lstm_model(
            input_shape=(24, 8),
            forecast_horizon=12,
        )
        dummy_input = tf.random.normal((4, 24, 8))
        output = model(dummy_input, training=True)
        assert not tf.reduce_any(tf.math.is_nan(output))

    def test_accepts_dataclass_config(self):
        from src.config import CNNConfig, LSTMConfig, ForecastHeadConfig
        model = build_cnn_lstm_model(
            input_shape=(24, 8),
            forecast_horizon=12,
            cnn_config=CNNConfig(filters=[16, 32], kernel_sizes=[3, 3]),
            lstm_config=LSTMConfig(units=[16]),
            forecast_config=ForecastHeadConfig(hidden_dims=[16]),
        )
        dummy_input = tf.random.normal((2, 24, 8))
        output = model(dummy_input, training=False)
        assert output.shape == (2, 12)

    def test_accepts_none_configs_uses_defaults(self):
        model = build_cnn_lstm_model(
            input_shape=(24, 8),
            forecast_horizon=12,
        )
        assert model is not None
        assert model.built


class TestMutableDefaults:
    def test_cnn_no_shared_mutable_defaults(self):
        a = CNNFeatureExtractor()
        b = CNNFeatureExtractor()
        assert a.filters is not b.filters or isinstance(a.filters, tuple)

    def test_lstm_no_shared_mutable_defaults(self):
        a = LSTMTemporalEncoder()
        b = LSTMTemporalEncoder()
        assert a.units_list is not b.units_list or isinstance(a.units_list, tuple)

    def test_head_no_shared_mutable_defaults(self):
        a = ForecastingHead(horizon=12)
        b = ForecastingHead(horizon=12)
        assert a.hidden_dims is not b.hidden_dims or isinstance(a.hidden_dims, tuple)


class TestCNNFeatureExtractor:
    def test_mismatched_filters_kernel_sizes_raises(self):
        from src.exceptions import ModelBuildError
        with pytest.raises(ModelBuildError):
            CNNFeatureExtractor(filters=(64,), kernel_sizes=(3, 3))
