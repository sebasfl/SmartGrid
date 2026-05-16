"""Batch inference script for the CNN-LSTM energy forecasting model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from .config import Config
from .data.loading import filter_by_building_ids, load_building_split, load_parquet
from .data.sequences import create_sequences
from .evaluation.metrics import ForecastMetrics
from .exceptions import DataError, InsufficientDataError
from .models.cnn_lstm import build_cnn_lstm_model
from .utils.gpu import configure_gpu

logger = logging.getLogger(__name__)


def load_artifacts(model_dir: Path) -> tuple[tf.keras.Model, Config, StandardScaler]:
    """Load trained model, config, and scaler from a model directory."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise DataError(f"Config not found: {config_path}")
    config = Config.from_json(str(config_path))

    scaler_path = model_dir / "scaler.pkl"
    if not scaler_path.exists():
        raise DataError(f"Scaler not found: {scaler_path}")

    import sklearn

    scaler_artifact = joblib.load(scaler_path)
    if isinstance(scaler_artifact, dict):
        scaler = scaler_artifact["scaler"]
        saved_version = scaler_artifact.get("sklearn_version", "unknown")
        if saved_version != sklearn.__version__:
            logger.warning(
                "Scaler saved with sklearn %s but current is %s — predictions may differ",
                saved_version,
                sklearn.__version__,
            )
    else:
        # Backwards compat: old format saved scaler directly
        scaler = scaler_artifact

    model = build_cnn_lstm_model(
        input_shape=(config.data.lookback_window, config.data.input_dim),
        forecast_horizon=config.data.forecast_horizon,
        cnn_config=config.cnn,
        lstm_config=config.lstm,
        forecast_config=config.forecast_head,
    )
    model.compile(loss=config.loss.forecast_loss_type, metrics=["mae", "mse"])

    weights_path = find_weights(model_dir)
    model.load_weights(str(weights_path))
    logger.info("Loaded weights from %s", weights_path)

    return model, config, scaler


def find_weights(model_dir: Path) -> Path:
    """Find the best or final model weights in a directory."""
    best_weights = sorted(model_dir.glob("best_model_epoch_*.h5"), reverse=True)
    if best_weights:
        return best_weights[0]

    final_weights = model_dir / "cnn_lstm_final.h5"
    if final_weights.exists():
        return final_weights

    raise DataError(f"No model weights found in {model_dir}")


def load_test_building_ids(building_split_path: Path) -> list[str]:
    """Load test building IDs from a building split JSON file."""
    split = load_building_split(str(building_split_path))
    test_ids = split.get("test", [])
    if not test_ids:
        raise InsufficientDataError("No test buildings found in building split file")
    return test_ids


def predict_building(
    df: pd.DataFrame,
    building_id: str,
    model: tf.keras.Model,
    scaler: StandardScaler,
    config: Config,
    batch_size: int,
) -> dict | None:
    """Run inference for a single building and return predictions with metrics."""
    X, y_true = create_sequences(
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
    X_normalized = X_normalized.reshape(original_shape).astype(np.float32)

    y_pred = model.predict(X_normalized, batch_size=batch_size, verbose=0)

    metrics = ForecastMetrics.compute_all(y_true, y_pred)

    return {
        "building_id": building_id,
        "n_sequences": len(X),
        "y_true": y_true,
        "y_pred": y_pred,
        "metrics": metrics,
    }


def predict(args: argparse.Namespace) -> None:
    """Main prediction pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    configure_gpu()

    model_dir = Path(args.model_dir)
    model, config, scaler = load_artifacts(model_dir)
    logger.info("Model loaded with %s parameters", f"{model.count_params():,}")

    df = load_parquet(args.parquet)

    if args.building_split:
        building_ids = load_test_building_ids(Path(args.building_split))
        df = filter_by_building_ids(df, building_ids)
        logger.info("Filtered to %d test buildings", len(building_ids))
    else:
        building_ids = df["building_id"].unique().tolist()
        logger.info("Using all %d buildings", len(building_ids))

    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    per_building_results: list[dict] = []
    skipped = 0

    for i, bid in enumerate(building_ids):
        result = predict_building(df, bid, model, scaler, config, args.batch_size)
        if result is None:
            skipped += 1
            continue

        all_y_true.append(result["y_true"])
        all_y_pred.append(result["y_pred"])
        per_building_results.append(
            {"building_id": result["building_id"], "n_sequences": result["n_sequences"], **result["metrics"]}
        )

        if (i + 1) % 50 == 0:
            logger.info("  Processed %d/%d buildings", i + 1, len(building_ids))

    if not per_building_results:
        raise InsufficientDataError("No buildings had enough data to generate predictions")

    logger.info("Predicted %d buildings, skipped %d (insufficient data)", len(per_building_results), skipped)

    aggregate_y_true = np.concatenate(all_y_true)
    aggregate_y_pred = np.concatenate(all_y_pred)
    aggregate_metrics = ForecastMetrics.compute_all(aggregate_y_true, aggregate_y_pred)

    results_df = pd.DataFrame(per_building_results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info("Per-building results saved to %s", output_path)

    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"  Buildings evaluated: {len(per_building_results)}")
    print(f"  Buildings skipped:   {skipped}")
    print(f"  Total sequences:     {int(results_df['n_sequences'].sum()):,}")
    print()
    print("  Aggregate Metrics:")
    for name, value in aggregate_metrics.items():
        print(f"    {name:>8s}: {value:.4f}")
    print()
    print("  Per-Building Averages:")
    for col in ["rmse", "mae", "mape", "smape", "r2", "nrmse"]:
        if col in results_df.columns:
            print(f"    {col:>8s}: {results_df[col].mean():.4f} (+/- {results_df[col].std():.4f})")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference for CNN-LSTM energy forecasting model")

    parser.add_argument(
        "--model_dir", type=str, required=True, help="Directory containing model weights, config.json, and scaler.pkl"
    )
    parser.add_argument("--parquet", type=str, required=True, help="Path to preprocessed parquet file")
    parser.add_argument("--building_split", type=str, default=None, help="Path to building_split.json (uses test set)")
    parser.add_argument(
        "--output", type=str, default="predictions.csv", help="Output CSV path for per-building results"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")

    predict(parser.parse_args())
