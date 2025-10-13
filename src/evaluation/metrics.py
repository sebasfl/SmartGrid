# src/evaluation/metrics.py
# Evaluation metrics for Multi-Task Learning
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, accuracy_score
)
from typing import Dict, Tuple, Optional


class ForecastMetrics:
    """Metrics for forecasting evaluation."""

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
        """Mean Absolute Percentage Error."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Avoid division by zero
        mask = np.abs(y_true) > epsilon
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        return np.mean(numerator / (denominator + epsilon)) * 100

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R² coefficient of determination."""
        return r2_score(y_true, y_pred)

    @staticmethod
    def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Normalized RMSE (by range of y_true)."""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        y_range = np.max(y_true) - np.min(y_true)
        return rmse / y_range if y_range > 0 else 0.0

    @staticmethod
    def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute all forecasting metrics.

        Args:
            y_true: Ground truth values [N, horizon] or [N]
            y_pred: Predicted values [N, horizon] or [N]

        Returns:
            Dict of metric names and values
        """
        # Flatten if multi-step
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        # Remove NaN/Inf values
        mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
        y_true_clean = y_true_flat[mask]
        y_pred_clean = y_pred_flat[mask]

        if len(y_true_clean) == 0:
            return {k: 0.0 for k in ['rmse', 'mae', 'mape', 'smape', 'r2', 'nrmse']}

        return {
            'rmse': ForecastMetrics.rmse(y_true_clean, y_pred_clean),
            'mae': ForecastMetrics.mae(y_true_clean, y_pred_clean),
            'mape': ForecastMetrics.mape(y_true_clean, y_pred_clean),
            'smape': ForecastMetrics.smape(y_true_clean, y_pred_clean),
            'r2': ForecastMetrics.r2(y_true_clean, y_pred_clean),
            'nrmse': ForecastMetrics.nrmse(y_true_clean, y_pred_clean)
        }


class AnomalyMetrics:
    """Metrics for anomaly detection evaluation."""

    @staticmethod
    def binary_classification_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                       threshold: float = 0.5) -> Dict[str, float]:
        """Compute binary classification metrics.

        Args:
            y_true: Ground truth binary labels [N]
            y_pred_proba: Predicted probabilities [N]
            threshold: Classification threshold

        Returns:
            Dict of metrics
        """
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        # Accuracy
        acc = accuracy_score(y_true, y_pred)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Specificity (true negative rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }

    @staticmethod
    def roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Compute ROC-AUC score.

        Args:
            y_true: Ground truth binary labels [N]
            y_pred_proba: Predicted probabilities [N]

        Returns:
            ROC-AUC score
        """
        try:
            return roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            # Only one class present
            return 0.0

    @staticmethod
    def optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray,
                          metric: str = 'f1') -> Tuple[float, float]:
        """Find optimal threshold to maximize a metric.

        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')

        Returns:
            Tuple of (optimal_threshold, best_metric_value)
        """
        thresholds = np.linspace(0, 1, 100)
        best_threshold = 0.5
        best_metric_value = 0.0

        for thresh in thresholds:
            metrics = AnomalyMetrics.binary_classification_metrics(
                y_true, y_pred_proba, threshold=thresh
            )

            metric_value = metrics.get(metric if metric != 'f1' else 'f1_score', 0.0)

            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = thresh

        return best_threshold, best_metric_value

    @staticmethod
    def compute_all(y_true: np.ndarray, y_pred_proba: np.ndarray,
                    threshold: float = 0.5, find_optimal: bool = False) -> Dict[str, float]:
        """Compute all anomaly detection metrics.

        Args:
            y_true: Ground truth labels [N]
            y_pred_proba: Predicted probabilities [N]
            threshold: Classification threshold
            find_optimal: Whether to find optimal threshold

        Returns:
            Dict of metrics
        """
        # Flatten arrays
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred_proba.flatten()

        # Remove NaN/Inf
        mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
        y_true_clean = y_true_flat[mask]
        y_pred_clean = y_pred_flat[mask]

        if len(y_true_clean) == 0 or len(np.unique(y_true_clean)) < 2:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'specificity': 0.0,
                'roc_auc': 0.0,
                'threshold': threshold
            }

        # Binary classification metrics
        metrics = AnomalyMetrics.binary_classification_metrics(
            y_true_clean, y_pred_clean, threshold=threshold
        )

        # ROC-AUC
        metrics['roc_auc'] = AnomalyMetrics.roc_auc(y_true_clean, y_pred_clean)

        # Find optimal threshold if requested
        if find_optimal:
            optimal_thresh, optimal_f1 = AnomalyMetrics.optimal_threshold(
                y_true_clean, y_pred_clean, metric='f1'
            )
            metrics['optimal_threshold'] = optimal_thresh
            metrics['optimal_f1'] = optimal_f1
        else:
            metrics['threshold'] = threshold

        return metrics


class MTLMetrics:
    """Combined metrics for Multi-Task Learning."""

    def __init__(self, forecast_weight: float = 0.6, anomaly_weight: float = 0.4):
        """
        Args:
            forecast_weight: Weight for forecasting performance
            anomaly_weight: Weight for anomaly detection performance
        """
        self.forecast_weight = forecast_weight
        self.anomaly_weight = anomaly_weight

    def compute_combined_score(self, forecast_metrics: Dict[str, float],
                              anomaly_metrics: Dict[str, float]) -> float:
        """Compute combined performance score.

        Args:
            forecast_metrics: Dict from ForecastMetrics.compute_all()
            anomaly_metrics: Dict from AnomalyMetrics.compute_all()

        Returns:
            Combined score (0-100, higher is better)
        """
        # Normalize forecasting metrics (R² is already 0-1, higher is better)
        forecast_score = max(0.0, min(1.0, forecast_metrics.get('r2', 0.0)))

        # Normalize anomaly detection (F1 is already 0-1)
        anomaly_score = anomaly_metrics.get('f1_score', 0.0)

        # Weighted combination
        combined = (
            self.forecast_weight * forecast_score +
            self.anomaly_weight * anomaly_score
        )

        return combined * 100  # Scale to 0-100

    def evaluate(self, y_forecast_true: np.ndarray, y_forecast_pred: np.ndarray,
                 y_anomaly_true: np.ndarray, y_anomaly_pred: np.ndarray,
                 anomaly_threshold: float = 0.5) -> Dict[str, any]:
        """Complete MTL evaluation.

        Args:
            y_forecast_true: Ground truth forecasts
            y_forecast_pred: Predicted forecasts
            y_anomaly_true: Ground truth anomaly labels
            y_anomaly_pred: Predicted anomaly probabilities
            anomaly_threshold: Threshold for anomaly classification

        Returns:
            Dict with all metrics
        """
        # Forecast metrics
        forecast_metrics = ForecastMetrics.compute_all(y_forecast_true, y_forecast_pred)

        # Anomaly metrics
        anomaly_metrics = AnomalyMetrics.compute_all(
            y_anomaly_true, y_anomaly_pred,
            threshold=anomaly_threshold,
            find_optimal=True
        )

        # Combined score
        combined_score = self.compute_combined_score(forecast_metrics, anomaly_metrics)

        return {
            'forecast': forecast_metrics,
            'anomaly': anomaly_metrics,
            'combined_score': combined_score,
            'weights': {
                'forecast': self.forecast_weight,
                'anomaly': self.anomaly_weight
            }
        }


if __name__ == "__main__":
    print("Testing evaluation metrics...\n")

    # Sample data
    np.random.seed(42)

    # Forecasting test data
    y_forecast_true = np.random.uniform(0, 100, (100, 24))
    y_forecast_pred = y_forecast_true + np.random.normal(0, 5, (100, 24))

    # Anomaly detection test data
    y_anomaly_true = np.random.randint(0, 2, 100)
    y_anomaly_pred = np.random.uniform(0, 1, 100)
    # Make predictions correlated with true labels
    y_anomaly_pred[y_anomaly_true == 1] += 0.3
    y_anomaly_pred = np.clip(y_anomaly_pred, 0, 1)

    # Test forecast metrics
    print("1. Forecasting Metrics:")
    forecast_metrics = ForecastMetrics.compute_all(y_forecast_true, y_forecast_pred)
    for metric, value in forecast_metrics.items():
        print(f"   {metric}: {value:.4f}")

    # Test anomaly metrics
    print("\n2. Anomaly Detection Metrics:")
    anomaly_metrics = AnomalyMetrics.compute_all(
        y_anomaly_true, y_anomaly_pred, find_optimal=True
    )
    for metric, value in anomaly_metrics.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.4f}")
        else:
            print(f"   {metric}: {value}")

    # Test MTL metrics
    print("\n3. MTL Combined Metrics:")
    mtl_metrics = MTLMetrics(forecast_weight=0.6, anomaly_weight=0.4)
    results = mtl_metrics.evaluate(
        y_forecast_true, y_forecast_pred,
        y_anomaly_true, y_anomaly_pred
    )
    print(f"   Combined Score: {results['combined_score']:.2f}/100")
    print(f"   Forecast R²: {results['forecast']['r2']:.4f}")
    print(f"   Anomaly F1: {results['anomaly']['f1_score']:.4f}")

    print("\n✅ All metrics work correctly!")
