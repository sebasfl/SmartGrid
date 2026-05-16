from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from ..config import Config

if TYPE_CHECKING:
    import tensorflow as tf

logger = logging.getLogger(__name__)


class ForecastMetrics:
    """Metrics for forecasting evaluation."""

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = np.abs(y_true) > epsilon
        if not mask.any():
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        return float(np.mean(numerator / (denominator + epsilon)) * 100)

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(r2_score(y_true, y_pred))

    @staticmethod
    def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
        y_range = np.max(y_true) - np.min(y_true)
        return float(rmse_val / y_range) if y_range > 0 else 0.0

    @staticmethod
    def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Compute all forecasting metrics, filtering out NaN/Inf values."""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
        y_true_clean = y_true_flat[mask]
        y_pred_clean = y_pred_flat[mask]

        if len(y_true_clean) == 0:
            return {k: 0.0 for k in ["rmse", "mae", "mape", "smape", "r2", "nrmse"]}

        return {
            "rmse": ForecastMetrics.rmse(y_true_clean, y_pred_clean),
            "mae": ForecastMetrics.mae(y_true_clean, y_pred_clean),
            "mape": ForecastMetrics.mape(y_true_clean, y_pred_clean),
            "smape": ForecastMetrics.smape(y_true_clean, y_pred_clean),
            "r2": ForecastMetrics.r2(y_true_clean, y_pred_clean),
            "nrmse": ForecastMetrics.nrmse(y_true_clean, y_pred_clean),
        }


class BuildingEvaluator:
    """Evaluate CNN-LSTM model performance on individual buildings."""

    @staticmethod
    def evaluate_single_building(
        df: pd.DataFrame,
        building_id: str,
        model: tf.keras.Model,
        scaler: StandardScaler,
        config: Config,
        create_sequences_fn: Callable,
    ) -> dict[str, float | str | int] | None:
        """Evaluate model on a single building. Returns None if insufficient data."""
        X, y_forecast = create_sequences_fn(
            df,
            building_id,
            lookback=config.data.lookback_window,
            horizon=config.data.forecast_horizon,
            stride=config.data.stride,
        )

        if X is None or len(X) == 0:
            return None

        original_shape = X.shape
        X_normalized = scaler.transform(X.reshape(-1, X.shape[-1]))
        X = X_normalized.reshape(original_shape)

        forecast_pred = model.predict(X, verbose=0)
        metrics = ForecastMetrics.compute_all(y_forecast, forecast_pred)

        return {
            "building_id": building_id,
            "n_sequences": len(X),
            "forecast_rmse": metrics["rmse"],
            "forecast_mae": metrics["mae"],
            "forecast_mape": metrics["mape"],
            "forecast_r2": metrics["r2"],
        }

    @staticmethod
    def evaluate_building_set(
        df: pd.DataFrame,
        building_ids: list[str],
        model: tf.keras.Model,
        scaler: StandardScaler,
        config: Config,
        create_sequences_fn: Callable,
        set_name: str = "test",
    ) -> pd.DataFrame:
        results: list[dict] = []
        skipped = 0

        for building_id in building_ids:
            metrics = BuildingEvaluator.evaluate_single_building(
                df,
                building_id,
                model,
                scaler,
                config,
                create_sequences_fn,
            )
            if metrics:
                results.append(metrics)
            else:
                skipped += 1

        if skipped > 0:
            logger.info("Skipped %d/%d buildings with insufficient data", skipped, len(building_ids))

        return pd.DataFrame(results)

    @staticmethod
    def save_evaluation_results(results_df: pd.DataFrame, output_dir: str, set_name: str = "test") -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_path = output_path / f"{set_name}_evaluation_results.csv"
        results_df.to_csv(results_path, index=False)

        summary = {
            "evaluation_type": f"{set_name}_set",
            "n_buildings": len(results_df),
            "total_sequences": int(results_df["n_sequences"].sum()),
            "forecast_metrics": {
                "rmse_mean": float(results_df["forecast_rmse"].mean()),
                "rmse_std": float(results_df["forecast_rmse"].std()),
                "rmse_median": float(results_df["forecast_rmse"].median()),
                "mae_mean": float(results_df["forecast_mae"].mean()),
                "mae_std": float(results_df["forecast_mae"].std()),
                "mape_mean": float(results_df["forecast_mape"].mean()),
                "mape_std": float(results_df["forecast_mape"].std()),
                "r2_mean": float(results_df["forecast_r2"].mean()),
                "r2_std": float(results_df["forecast_r2"].std()),
            },
        }

        summary_path = output_path / f"{set_name}_evaluation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
