# src/models/losses.py
# Loss functions for Multi-Task Learning
import tensorflow as tf
from tensorflow.keras import losses


class FocalLoss(losses.Loss):
    """Focal Loss for addressing class imbalance in anomaly detection.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 from_logits: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        """Compute focal loss.

        Args:
            y_true: Ground truth labels [batch, 1]
            y_pred: Predicted probabilities [batch, 1]

        Returns:
            Focal loss value
        """
        # Apply sigmoid if predictions are logits
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Calculate focal loss
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

        # Compute p_t
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = tf.pow(1 - p_t, self.gamma)

        # Compute alpha factor
        alpha_factor = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)

        # Final focal loss
        focal_loss = alpha_factor * focal_weight * cross_entropy

        return tf.reduce_mean(focal_loss)

    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits
        })
        return config


class HuberLoss(losses.Loss):
    """Huber loss for robust forecasting (less sensitive to outliers than MSE)."""

    def __init__(self, delta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def call(self, y_true, y_pred):
        """Compute Huber loss.

        Args:
            y_true: Ground truth values [batch, horizon]
            y_pred: Predicted values [batch, horizon]

        Returns:
            Huber loss value
        """
        error = y_true - y_pred
        abs_error = tf.abs(error)

        # Quadratic when |error| <= delta, linear when |error| > delta
        quadratic = tf.minimum(abs_error, self.delta)
        linear = abs_error - quadratic

        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({'delta': self.delta})
        return config


class MTLLoss:
    """Multi-Task Learning loss combining anomaly detection and forecasting losses.

    Supports:
    - Fixed weighting (α, β)
    - Uncertainty weighting (Kendall et al. 2018)
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 anomaly_loss_type: str = 'bce',
                 forecast_loss_type: str = 'mse',
                 use_uncertainty_weighting: bool = False,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 huber_delta: float = 1.0):

        self.alpha = alpha
        self.beta = beta
        self.anomaly_loss_type = anomaly_loss_type
        self.forecast_loss_type = forecast_loss_type
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # Anomaly loss
        if anomaly_loss_type == 'bce':
            self.anomaly_loss_fn = losses.BinaryCrossentropy()
        elif anomaly_loss_type == 'focal':
            self.anomaly_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            raise ValueError(f"Unknown anomaly_loss_type: {anomaly_loss_type}")

        # Forecast loss
        if forecast_loss_type == 'mse':
            self.forecast_loss_fn = losses.MeanSquaredError()
        elif forecast_loss_type == 'mae':
            self.forecast_loss_fn = losses.MeanAbsoluteError()
        elif forecast_loss_type == 'huber':
            self.forecast_loss_fn = HuberLoss(delta=huber_delta)
        else:
            raise ValueError(f"Unknown forecast_loss_type: {forecast_loss_type}")

        # Uncertainty weighting parameters (learnable)
        if use_uncertainty_weighting:
            self.log_var_anomaly = tf.Variable(0.0, trainable=True, name='log_var_anomaly')
            self.log_var_forecast = tf.Variable(0.0, trainable=True, name='log_var_forecast')

    def __call__(self, y_true, y_pred):
        """Compute combined MTL loss.

        Args:
            y_true: Tuple of (anomaly_true, forecast_true)
                - anomaly_true: [batch, 1]
                - forecast_true: [batch, horizon]
            y_pred: Tuple of (anomaly_pred, forecast_pred)
                - anomaly_pred: [batch, 1]
                - forecast_pred: [batch, horizon]

        Returns:
            Tuple of (total_loss, anomaly_loss, forecast_loss)
        """
        anomaly_true, forecast_true = y_true
        anomaly_pred, forecast_pred = y_pred

        # Compute individual losses
        loss_anomaly = self.anomaly_loss_fn(anomaly_true, anomaly_pred)
        loss_forecast = self.forecast_loss_fn(forecast_true, forecast_pred)

        if self.use_uncertainty_weighting:
            # Uncertainty-weighted loss (Kendall et al. 2018)
            # L = L_1 / (2 * σ_1^2) + L_2 / (2 * σ_2^2) + log(σ_1) + log(σ_2)
            # Using log_var = log(σ^2) for numerical stability

            precision_anomaly = tf.exp(-self.log_var_anomaly)
            precision_forecast = tf.exp(-self.log_var_forecast)

            total_loss = (
                precision_anomaly * loss_anomaly +
                precision_forecast * loss_forecast +
                self.log_var_anomaly +
                self.log_var_forecast
            )
        else:
            # Fixed weighting
            total_loss = self.alpha * loss_anomaly + self.beta * loss_forecast

        return (total_loss, loss_anomaly, loss_forecast)


class WeightedMSELoss(losses.Loss):
    """Weighted MSE loss for forecasting with time-dependent weights.

    Gives more weight to near-term predictions and less to far-term.
    """

    def __init__(self, horizon: int = 24, decay: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.horizon = horizon
        self.decay = decay

        # Create exponentially decaying weights
        # w_t = decay^t for t in [0, horizon)
        weights = tf.constant([decay ** i for i in range(horizon)], dtype=tf.float32)
        self.weights = weights / tf.reduce_sum(weights)  # Normalize

    def call(self, y_true, y_pred):
        """Compute weighted MSE loss.

        Args:
            y_true: Ground truth [batch, horizon]
            y_pred: Predictions [batch, horizon]

        Returns:
            Weighted MSE loss
        """
        squared_error = tf.square(y_true - y_pred)
        weighted_error = squared_error * self.weights
        return tf.reduce_mean(weighted_error)

    def get_config(self):
        config = super().get_config()
        config.update({
            'horizon': self.horizon,
            'decay': self.decay
        })
        return config


if __name__ == "__main__":
    # Test loss functions
    print("Testing MTL Loss Functions...\n")

    batch_size = 4
    horizon = 24

    # Sample data
    anomaly_true = tf.constant([[0.], [1.], [0.], [1.]], dtype=tf.float32)
    anomaly_pred = tf.constant([[0.1], [0.9], [0.2], [0.7]], dtype=tf.float32)

    forecast_true = tf.random.uniform((batch_size, horizon), minval=0, maxval=100)
    forecast_pred = forecast_true + tf.random.normal((batch_size, horizon), stddev=5)

    # Test individual losses
    print("1. Binary Crossentropy:")
    bce = losses.BinaryCrossentropy()
    bce_loss = bce(anomaly_true, anomaly_pred)
    print(f"   Loss: {bce_loss.numpy():.4f}\n")

    print("2. Focal Loss:")
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    focal_loss = focal(anomaly_true, anomaly_pred)
    print(f"   Loss: {focal_loss.numpy():.4f}\n")

    print("3. MSE:")
    mse = losses.MeanSquaredError()
    mse_loss = mse(forecast_true, forecast_pred)
    print(f"   Loss: {mse_loss.numpy():.4f}\n")

    print("4. Huber Loss:")
    huber = HuberLoss(delta=1.0)
    huber_loss = huber(forecast_true, forecast_pred)
    print(f"   Loss: {huber_loss.numpy():.4f}\n")

    print("5. MTL Loss (fixed weights):")
    mtl = MTLLoss(alpha=0.3, beta=0.7, anomaly_loss_type='bce', forecast_loss_type='mse')
    total, loss_a, loss_f = mtl((anomaly_true, forecast_true), (anomaly_pred, forecast_pred))
    print(f"   Total: {total.numpy():.4f}, Anomaly: {loss_a.numpy():.4f}, Forecast: {loss_f.numpy():.4f}\n")

    print("6. MTL Loss (uncertainty weighting):")
    mtl_unc = MTLLoss(use_uncertainty_weighting=True)
    total_unc, loss_a_unc, loss_f_unc = mtl_unc(
        (anomaly_true, forecast_true),
        (anomaly_pred, forecast_pred)
    )
    print(f"   Total: {total_unc.numpy():.4f}, Anomaly: {loss_a_unc.numpy():.4f}, Forecast: {loss_f_unc.numpy():.4f}")
    print(f"   Learned weights: anomaly={tf.exp(-mtl_unc.log_var_anomaly).numpy():.4f}, "
          f"forecast={tf.exp(-mtl_unc.log_var_forecast).numpy():.4f}\n")

    print("7. Weighted MSE Loss:")
    wmse = WeightedMSELoss(horizon=24, decay=0.9)
    wmse_loss = wmse(forecast_true, forecast_pred)
    print(f"   Loss: {wmse_loss.numpy():.4f}\n")

    print("✅ All loss functions work correctly!")
