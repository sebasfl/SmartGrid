# src/training/trainer.py
# Multi-Task Learning Trainer with GPU optimization
import tensorflow as tf
from tensorflow.keras import optimizers, mixed_precision
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np


class MTLTrainer:
    """Multi-Task Learning trainer with GPU support and mixed precision."""

    def __init__(self, model, loss_fn, optimizer_config, training_config, metrics=None):
        """Initialize MTL trainer.

        Args:
            model: MTL model (encoder + heads)
            loss_fn: MTL loss function
            optimizer_config: Optimizer configuration from config.py
            training_config: Training configuration from config.py
            metrics: Optional dict of metric functions
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_config = optimizer_config
        self.training_config = training_config
        self.metrics = metrics or {}

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Mixed precision setup
        if training_config.use_mixed_precision:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer)
            print("✅ Mixed precision (FP16) enabled")

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'train_anomaly_loss': [],
            'train_forecast_loss': [],
            'val_loss': [],
            'val_anomaly_loss': [],
            'val_forecast_loss': [],
            'learning_rate': []
        }

    def _create_optimizer(self):
        """Create optimizer from config."""
        opt_config = self.optimizer_config

        if opt_config.optimizer_type == 'adam':
            optimizer = optimizers.Adam(
                learning_rate=opt_config.learning_rate,
                beta_1=opt_config.beta1,
                beta_2=opt_config.beta2,
                epsilon=opt_config.epsilon
            )
        elif opt_config.optimizer_type == 'adamw':
            optimizer = optimizers.AdamW(
                learning_rate=opt_config.learning_rate,
                weight_decay=opt_config.weight_decay,
                beta_1=opt_config.beta1,
                beta_2=opt_config.beta2
            )
        elif opt_config.optimizer_type == 'sgd':
            optimizer = optimizers.SGD(
                learning_rate=opt_config.learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.optimizer_type}")

        return optimizer

    # @tf.function  # Disabled for compatibility with Python generators
    def train_step(self, x, y_anomaly, y_forecast):
        """Single training step with gradient computation.

        Args:
            x: Input sequences [batch, seq_len, features]
            y_anomaly: Anomaly labels [batch, 1]
            y_forecast: Forecast targets [batch, horizon]

        Returns:
            Tuple of (total_loss, anomaly_loss, forecast_loss)
        """
        with tf.GradientTape() as tape:
            # Forward pass
            anomaly_pred, forecast_pred = self.model(x, training=True)

            # Compute loss
            total_loss, loss_anomaly, loss_forecast = self.loss_fn(
                (y_anomaly, y_forecast),
                (anomaly_pred, forecast_pred)
            )

            # Scale loss for mixed precision
            if self.training_config.use_mixed_precision:
                scaled_loss = self.optimizer.get_scaled_loss(total_loss)
            else:
                scaled_loss = total_loss

        # Compute gradients
        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)

        # Unscale gradients if using mixed precision
        if self.training_config.use_mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)

        # Clip gradients
        gradients, _ = tf.clip_by_global_norm(
            gradients,
            self.optimizer_config.gradient_clip_norm
        )

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return total_loss, loss_anomaly, loss_forecast

    # @tf.function  # Disabled for compatibility with Python generators
    def val_step(self, x, y_anomaly, y_forecast):
        """Single validation step.

        Args:
            x: Input sequences [batch, seq_len, features]
            y_anomaly: Anomaly labels [batch, 1]
            y_forecast: Forecast targets [batch, horizon]

        Returns:
            Tuple of (total_loss, anomaly_loss, forecast_loss)
        """
        # Forward pass (no training mode)
        anomaly_pred, forecast_pred = self.model(x, training=False)

        # Compute loss
        total_loss, loss_anomaly, loss_forecast = self.loss_fn(
            (y_anomaly, y_forecast),
            (anomaly_pred, forecast_pred)
        )

        return total_loss, loss_anomaly, loss_forecast

    def train_epoch(self, train_dataset):
        """Train for one epoch.

        Args:
            train_dataset: TensorFlow dataset for training

        Returns:
            Dict of epoch metrics
        """
        epoch_losses = {'total': [], 'anomaly': [], 'forecast': []}

        start_time = time.time()
        num_batches = 0

        for batch_idx, (x, y_anomaly, y_forecast) in enumerate(train_dataset):
            # Training step
            total_loss, loss_a, loss_f = self.train_step(x, y_anomaly, y_forecast)

            # Check for NaN in loss (should not happen with cleaned data)
            if tf.math.is_nan(total_loss):
                print(f"\n❌ ERROR: NaN loss detected at batch {batch_idx + 1}!")
                print(f"   Anomaly loss: {loss_a}, Forecast loss: {loss_f}")
                print(f"   This indicates a numerical instability issue.")
                print(f"   Stopping training...")
                raise ValueError("Training stopped due to NaN loss")

            # Record losses
            epoch_losses['total'].append(float(total_loss))
            epoch_losses['anomaly'].append(float(loss_a))
            epoch_losses['forecast'].append(float(loss_f))

            self.global_step += 1
            num_batches += 1

            # Log progress
            if (batch_idx + 1) % self.training_config.log_freq == 0:
                avg_loss = np.mean(epoch_losses['total'][-self.training_config.log_freq:])
                elapsed = time.time() - start_time
                batches_per_sec = num_batches / elapsed
                print(f"  Batch {batch_idx + 1}: Loss={avg_loss:.4f} ({batches_per_sec:.1f} batch/s)")

        # Compute epoch averages
        metrics = {
            'loss': np.mean(epoch_losses['total']),
            'anomaly_loss': np.mean(epoch_losses['anomaly']),
            'forecast_loss': np.mean(epoch_losses['forecast']),
            'time': time.time() - start_time
        }

        return metrics

    def validate(self, val_dataset):
        """Validate on validation set.

        Args:
            val_dataset: TensorFlow dataset for validation

        Returns:
            Dict of validation metrics
        """
        val_losses = {'total': [], 'anomaly': [], 'forecast': []}

        for x, y_anomaly, y_forecast in val_dataset:
            total_loss, loss_a, loss_f = self.val_step(x, y_anomaly, y_forecast)

            val_losses['total'].append(float(total_loss))
            val_losses['anomaly'].append(float(loss_a))
            val_losses['forecast'].append(float(loss_f))

        metrics = {
            'loss': np.mean(val_losses['total']),
            'anomaly_loss': np.mean(val_losses['anomaly']),
            'forecast_loss': np.mean(val_losses['forecast'])
        }

        return metrics

    def fit(self, train_dataset, val_dataset, callbacks=None):
        """Training loop with callbacks.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            callbacks: List of callback objects

        Returns:
            Training history dict
        """
        callbacks = callbacks or []
        epochs = self.training_config.epochs

        print(f"\n{'='*70}")
        print(f"🚀 STARTING MTL TRAINING")
        print(f"{'='*70}")
        print(f"Epochs: {epochs}")
        print(f"Optimizer: {self.optimizer_config.optimizer_type}")
        print(f"Learning rate: {self.optimizer_config.learning_rate}")
        print(f"Mixed precision: {self.training_config.use_mixed_precision}")
        print(f"{'='*70}\n")

        # Initialize callbacks
        for callback in callbacks:
            callback.on_train_begin()

        try:
            for epoch in range(epochs):
                self.epoch = epoch
                epoch_start = time.time()

                print(f"\nEpoch {epoch + 1}/{epochs}")
                print("-" * 50)

                # Train
                train_metrics = self.train_epoch(train_dataset)

                # Validate
                if (epoch + 1) % self.training_config.validation_freq == 0:
                    val_metrics = self.validate(val_dataset)
                else:
                    val_metrics = None

                # Record history
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['train_anomaly_loss'].append(train_metrics['anomaly_loss'])
                self.history['train_forecast_loss'].append(train_metrics['forecast_loss'])

                if val_metrics:
                    self.history['val_loss'].append(val_metrics['loss'])
                    self.history['val_anomaly_loss'].append(val_metrics['anomaly_loss'])
                    self.history['val_forecast_loss'].append(val_metrics['forecast_loss'])

                # Get current learning rate
                if isinstance(self.optimizer, mixed_precision.LossScaleOptimizer):
                    lr = float(self.optimizer.inner_optimizer.learning_rate)
                else:
                    lr = float(self.optimizer.learning_rate)
                self.history['learning_rate'].append(lr)

                # Print epoch summary
                epoch_time = time.time() - epoch_start
                print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
                      f"Anomaly: {train_metrics['anomaly_loss']:.4f}, "
                      f"Forecast: {train_metrics['forecast_loss']:.4f}")

                if val_metrics:
                    print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                          f"Anomaly: {val_metrics['anomaly_loss']:.4f}, "
                          f"Forecast: {val_metrics['forecast_loss']:.4f}")

                print(f"Time: {epoch_time:.2f}s, LR: {lr:.6f}")

                # Callbacks
                stop_training = False
                for callback in callbacks:
                    metrics_dict = {**train_metrics}
                    if val_metrics:
                        metrics_dict.update({f'val_{k}': v for k, v in val_metrics.items()})

                    if callback.on_epoch_end(epoch, metrics_dict, self.model, self.optimizer):
                        stop_training = True

                if stop_training:
                    print(f"\n⚠️  Training stopped by callback at epoch {epoch + 1}")
                    break

        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")

        finally:
            # Finalize callbacks
            for callback in callbacks:
                callback.on_train_end()

        print(f"\n{'='*70}")
        print(f"✅ TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Total epochs: {self.epoch + 1}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")

        return self.history

    def save_history(self, filepath: str):
        """Save training history to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"📊 Training history saved to {filepath}")

    def load_history(self, filepath: str):
        """Load training history from JSON."""
        with open(filepath, 'r') as f:
            self.history = json.load(f)
        print(f"📊 Training history loaded from {filepath}")


if __name__ == "__main__":
    print("Testing MTLTrainer...")

    # Create dummy model and data
    from ..models.transformer import TransformerEncoder
    from ..models.mtl_heads import MTLModel, AnomalyDetectionHead, ForecastingHead
    from ..models.losses import MTLLoss
    from ..config import Config

    config = Config()

    # Build model
    encoder = TransformerEncoder(
        num_layers=2,
        d_model=64,
        num_heads=4,
        ff_dim=128,
        input_dim=8,
        dropout=0.1
    )
    anomaly_head = AnomalyDetectionHead(hidden_dims=[32])
    forecast_head = ForecastingHead(forecast_horizon=24, hidden_dims=[64])
    model = MTLModel(encoder, anomaly_head, forecast_head)

    # Create trainer
    loss_fn = MTLLoss(alpha=0.3, beta=0.7)
    trainer = MTLTrainer(model, loss_fn, config.optimizer, config.training)

    # Create dummy dataset
    x = tf.random.normal((100, 168, 8))
    y_anomaly = tf.random.uniform((100, 1), maxval=2, dtype=tf.int32)
    y_forecast = tf.random.uniform((100, 24))

    dataset = tf.data.Dataset.from_tensor_slices((x, y_anomaly, y_forecast)).batch(32)

    print("✅ MTLTrainer initialized successfully!")
