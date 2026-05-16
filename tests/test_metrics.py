import numpy as np

from src.evaluation.metrics import ForecastMetrics


class TestForecastMetrics:
    def test_rmse_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert ForecastMetrics.rmse(y, y) == 0.0

    def test_mae_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert ForecastMetrics.mae(y, y) == 0.0

    def test_r2_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert ForecastMetrics.r2(y, y) == 1.0

    def test_rmse_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert abs(ForecastMetrics.rmse(y_true, y_pred) - 1.0) < 1e-7

    def test_mape_avoids_division_by_zero(self):
        y_true = np.array([0.0, 0.0, 1.0])
        y_pred = np.array([0.5, 0.5, 1.5])
        # Should not raise, only computes for non-zero values
        result = ForecastMetrics.mape(y_true, y_pred)
        assert result >= 0

    def test_compute_all_returns_expected_keys(self):
        y_true = np.random.uniform(1, 100, (50, 12))
        y_pred = y_true + np.random.normal(0, 1, (50, 12))
        result = ForecastMetrics.compute_all(y_true, y_pred)

        expected_keys = {'rmse', 'mae', 'mape', 'smape', 'r2', 'nrmse'}
        assert set(result.keys()) == expected_keys

    def test_compute_all_handles_nan(self):
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.0, 2.0, np.nan])
        result = ForecastMetrics.compute_all(y_true, y_pred)
        assert all(isinstance(v, float) for v in result.values())

    def test_compute_all_empty_after_filtering(self):
        y_true = np.array([np.nan])
        y_pred = np.array([np.nan])
        result = ForecastMetrics.compute_all(y_true, y_pred)
        assert all(v == 0.0 for v in result.values())

    def test_nrmse_zero_range(self):
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([6.0, 6.0, 6.0])
        assert ForecastMetrics.nrmse(y_true, y_pred) == 0.0
