from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision, optimizers

from ..config import OptimizerConfig, TrainingConfig
from ..exceptions import TrainingError

logger = logging.getLogger(__name__)


class CNNLSTMTrainer:
    """CNN-LSTM trainer with mixed precision and gradient clipping."""

    def __init__(
        self,
        model: tf.keras.Model,
        loss_fn: tf.keras.losses.Loss,
        optimizer_config: OptimizerConfig,
        training_config: TrainingConfig,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_config = optimizer_config
        self.training_config = training_config
        self.metrics = metrics or {}

        self.optimizer = self._create_optimizer()

        if training_config.use_mixed_precision:
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_global_policy(policy)
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer)
            logger.info("Mixed precision (FP16) enabled")

        self.epoch: int = 0
        self.global_step: int = 0
        self.best_val_loss: float = float("inf")
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

    def _create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        cfg = self.optimizer_config

        if cfg.optimizer_type == "adam":
            return optimizers.Adam(
                learning_rate=cfg.learning_rate,
                beta_1=cfg.beta1,
                beta_2=cfg.beta2,
                epsilon=cfg.epsilon,
            )
        elif cfg.optimizer_type == "adamw":
            return optimizers.AdamW(
                learning_rate=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                beta_1=cfg.beta1,
                beta_2=cfg.beta2,
            )
        elif cfg.optimizer_type == "sgd":
            return optimizers.SGD(learning_rate=cfg.learning_rate, momentum=0.9)
        else:
            raise TrainingError(f"Unknown optimizer: {cfg.optimizer_type}")

    # @tf.function disabled for cuDNN 9.0+ compatibility
    def train_step(self, x: tf.Tensor, y_forecast: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            forecast_pred = self.model(x, training=True)
            loss = self.loss_fn(y_forecast, forecast_pred)

            if self.training_config.use_mixed_precision:
                scaled_loss = self.optimizer.get_scaled_loss(loss)
            else:
                scaled_loss = loss

        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)

        if self.training_config.use_mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)

        gradients, _ = tf.clip_by_global_norm(gradients, self.optimizer_config.gradient_clip_norm)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables, strict=False))

        return loss

    # @tf.function disabled for cuDNN 9.0+ compatibility
    def val_step(self, x: tf.Tensor, y_forecast: tf.Tensor) -> tf.Tensor:
        forecast_pred = self.model(x, training=False)
        return self.loss_fn(y_forecast, forecast_pred)

    def train_epoch(self, train_dataset: tf.data.Dataset) -> dict[str, float]:
        epoch_losses: list[float] = []
        start_time = time.time()
        num_batches = 0

        for batch_idx, (x, y_forecast) in enumerate(train_dataset):
            loss = self.train_step(x, y_forecast)

            if tf.math.is_nan(loss):
                raise TrainingError(
                    f"NaN loss at batch {batch_idx + 1}. Check data for extreme values or reduce learning rate."
                )

            epoch_losses.append(float(loss))
            self.global_step += 1
            num_batches += 1

            if (batch_idx + 1) % self.training_config.log_freq == 0:
                avg_loss = np.mean(epoch_losses[-self.training_config.log_freq :])
                batches_per_sec = num_batches / (time.time() - start_time)
                logger.debug("Batch %d: Loss=%.4f (%.1f batch/s)", batch_idx + 1, avg_loss, batches_per_sec)

        return {"loss": np.mean(epoch_losses), "time": time.time() - start_time}

    def validate(self, val_dataset: tf.data.Dataset) -> dict[str, float]:
        val_losses: list[float] = []
        for x, y_forecast in val_dataset:
            val_losses.append(float(self.val_step(x, y_forecast)))
        return {"loss": np.mean(val_losses)}

    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        callbacks: list[Any] | None = None,
    ) -> dict[str, list[float]]:
        callbacks = callbacks or []
        epochs = self.training_config.epochs

        logger.info("=" * 70)
        logger.info("STARTING CNN-LSTM TRAINING")
        logger.info(
            "  Epochs: %d | Optimizer: %s | LR: %s | Mixed precision: %s",
            epochs,
            self.optimizer_config.optimizer_type,
            self.optimizer_config.learning_rate,
            self.training_config.use_mixed_precision,
        )
        logger.info("=" * 70)

        for callback in callbacks:
            callback.on_train_begin()

        try:
            for epoch in range(epochs):
                self.epoch = epoch
                logger.info("Epoch %d/%d", epoch + 1, epochs)

                train_metrics = self.train_epoch(train_dataset)

                val_metrics: dict[str, float] | None = None
                if (epoch + 1) % self.training_config.validation_freq == 0:
                    val_metrics = self.validate(val_dataset)

                self.history["train_loss"].append(train_metrics["loss"])
                if val_metrics:
                    self.history["val_loss"].append(val_metrics["loss"])
                    if val_metrics["loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["loss"]

                if isinstance(self.optimizer, mixed_precision.LossScaleOptimizer):
                    lr = float(self.optimizer.inner_optimizer.learning_rate)
                else:
                    lr = float(self.optimizer.learning_rate)
                self.history["learning_rate"].append(lr)

                logger.info(
                    "  Train Loss: %.4f%s | LR: %.6f",
                    train_metrics["loss"],
                    f" | Val Loss: {val_metrics['loss']:.4f}" if val_metrics else "",
                    lr,
                )

                stop_training = False
                for callback in callbacks:
                    metrics_dict = {**train_metrics}
                    if val_metrics:
                        metrics_dict.update({f"val_{k}": v for k, v in val_metrics.items()})
                    if callback.on_epoch_end(epoch, metrics_dict, self.model, self.optimizer):
                        stop_training = True

                if stop_training:
                    logger.warning("Training stopped by callback at epoch %d", epoch + 1)
                    break

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")

        finally:
            for callback in callbacks:
                callback.on_train_end()

        logger.info("=" * 70)
        logger.info("TRAINING COMPLETED — %d epochs, best val loss: %.4f", self.epoch + 1, self.best_val_loss)
        logger.info("=" * 70)

        return self.history

    def save_history(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info("Training history saved to %s", filepath)

    def load_history(self, filepath: str) -> None:
        with open(filepath) as f:
            self.history = json.load(f)
        logger.info("Training history loaded from %s", filepath)
