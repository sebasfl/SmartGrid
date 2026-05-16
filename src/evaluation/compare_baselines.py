from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression

from .metrics import ForecastMetrics

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    name: str
    metrics: dict[str, float]
    predictions: np.ndarray


def persistence_forecast(X: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed value for the entire horizon."""
    last_values = X[:, -1, -1]
    return np.tile(last_values[:, np.newaxis], (1, horizon))


def seasonal_persistence_forecast(X: np.ndarray, horizon: int, season_length: int = 8) -> np.ndarray:
    """Use values from the end of the lookback window, cycling seasonally."""
    n_samples = X.shape[0]
    predictions = np.zeros((n_samples, horizon), dtype=np.float32)
    values = X[:, :, -1]
    for i in range(horizon):
        source_idx = -(season_length - (i % season_length))
        predictions[:, i] = values[:, source_idx]
    return predictions


def linear_regression_forecast(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Fit linear regression on flattened sequences."""
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_features_flat = X_train.shape[1] * X_train.shape[2]

    X_train_flat = X_train.reshape(n_train, n_features_flat)
    X_test_flat = X_test.reshape(n_test, n_features_flat)

    model = LinearRegression()
    model.fit(X_train_flat, y_train)
    return model.predict(X_test_flat)  # type: ignore[no-any-return]


def compare_baselines(
    X_test: np.ndarray,
    y_test: np.ndarray,
    cnn_lstm_predictions: np.ndarray | None = None,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    season_length: int = 8,
) -> list[BaselineResult]:
    """Compare CNN-LSTM against baseline models.

    Args:
        X_test: Test input sequences [N, lookback, features].
        y_test: Test targets [N, horizon].
        cnn_lstm_predictions: Optional CNN-LSTM predictions [N, horizon].
        X_train: Training sequences (needed for linear regression).
        y_train: Training targets (needed for linear regression).
        season_length: Season length for seasonal persistence
            (default: 8 = 1 day at 3h intervals).

    Returns:
        List of BaselineResult sorted by RMSE (best first).
    """
    horizon = y_test.shape[1]
    results: list[BaselineResult] = []

    persist_pred = persistence_forecast(X_test, horizon)
    persist_metrics = ForecastMetrics.compute_all(y_test, persist_pred)
    results.append(BaselineResult("Persistence", persist_metrics, persist_pred))
    logger.info("Persistence RMSE: %.4f", persist_metrics["rmse"])

    seasonal_pred = seasonal_persistence_forecast(X_test, horizon, season_length)
    seasonal_metrics = ForecastMetrics.compute_all(y_test, seasonal_pred)
    results.append(BaselineResult("Seasonal Persistence", seasonal_metrics, seasonal_pred))
    logger.info("Seasonal Persistence RMSE: %.4f", seasonal_metrics["rmse"])

    if X_train is not None and y_train is not None:
        lr_pred = linear_regression_forecast(X_train, y_train, X_test)
        lr_metrics = ForecastMetrics.compute_all(y_test, lr_pred)
        results.append(BaselineResult("Linear Regression", lr_metrics, lr_pred))
        logger.info("Linear Regression RMSE: %.4f", lr_metrics["rmse"])

    if cnn_lstm_predictions is not None:
        cnn_metrics = ForecastMetrics.compute_all(y_test, cnn_lstm_predictions)
        results.append(BaselineResult("CNN-LSTM", cnn_metrics, cnn_lstm_predictions))
        logger.info("CNN-LSTM RMSE: %.4f", cnn_metrics["rmse"])

    results.sort(key=lambda r: r.metrics["rmse"])

    return results


def format_comparison_table(results: list[BaselineResult]) -> str:
    """Format results as a readable comparison table."""
    r2_label = "R\u00b2"
    header = f"{'Model':<25} {'RMSE':>10} {'MAE':>10} {'MAPE':>10} {r2_label:>10}"
    separator = "-" * len(header)
    lines = [separator, header, separator]

    for r in results:
        m = r.metrics
        lines.append(f"{r.name:<25} {m['rmse']:>10.4f} {m['mae']:>10.4f} {m['mape']:>10.2f} {m['r2']:>10.4f}")

    lines.append(separator)
    return "\n".join(lines)
