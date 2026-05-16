import numpy as np

from src.data.sequences import create_sequences


class TestCreateSequences:
    def test_returns_correct_shapes(self, sample_building_df):
        lookback, horizon, stride = 24, 12, 4
        X, y = create_sequences(sample_building_df, 'building_A', lookback, horizon, stride)

        assert X is not None
        assert y is not None
        assert X.shape[1] == lookback
        assert X.shape[2] == 8  # 7 time features + value
        assert y.shape[1] == horizon
        assert X.shape[0] == y.shape[0]

    def test_returns_none_when_data_too_short(self, sample_building_df):
        # lookback + horizon > data length
        X, y = create_sequences(sample_building_df, 'building_A', 500, 500, 1)
        assert X is None
        assert y is None

    def test_returns_none_for_missing_building(self, sample_building_df):
        X, y = create_sequences(sample_building_df, 'nonexistent', 24, 12, 1)
        assert X is None
        assert y is None

    def test_stride_affects_count(self, sample_building_df):
        X1, _ = create_sequences(sample_building_df, 'building_A', 24, 12, 1)
        X4, _ = create_sequences(sample_building_df, 'building_A', 24, 12, 4)

        assert X1 is not None
        assert X4 is not None
        # Stride 4 should produce roughly 1/4 the sequences
        assert X4.shape[0] < X1.shape[0]

    def test_output_dtype_is_float32(self, sample_building_df):
        X, y = create_sequences(sample_building_df, 'building_A', 24, 12, 4)
        assert X.dtype == np.float32
        assert y.dtype == np.float32

    def test_forecast_target_is_last_feature(self, sample_building_df):
        """Forecast target should be the 'value' column (last feature)."""
        X, y = create_sequences(sample_building_df, 'building_A', 24, 12, 4)
        # y should contain scalar values, not multi-feature vectors
        assert y.ndim == 2
        assert y.shape[1] == 12
