from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class Callback:
    """Base callback class."""

    def on_train_begin(self) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float],
                     model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer) -> bool:
        return False

    def on_train_end(self) -> None:
        pass


class EarlyStopping(Callback):
    """Stop training when monitored metric stops improving."""

    def __init__(self, monitor: str = 'val_loss', patience: int = 10,
                 min_delta: float = 1e-4, mode: str = 'min',
                 verbose: bool = True) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.wait: int = 0
        self.best_value: float = np.inf if mode == 'min' else -np.inf
        self.stopped_epoch: int = 0

    def on_train_begin(self) -> None:
        self.wait = 0
        self.best_value = np.inf if self.mode == 'min' else -np.inf
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch: int, metrics: dict[str, float],
                     model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer) -> bool:
        current_value = metrics.get(self.monitor)

        if current_value is None:
            if self.verbose:
                logger.warning("EarlyStopping: metric '%s' not found in metrics", self.monitor)
            return False

        if self.mode == 'min':
            improved = (self.best_value - current_value) > self.min_delta
        else:
            improved = (current_value - self.best_value) > self.min_delta

        if improved:
            self.best_value = current_value
            self.wait = 0
            if self.verbose:
                logger.info("EarlyStopping: %s improved to %.6f", self.monitor, current_value)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    logger.info("EarlyStopping: stopping at epoch %d", epoch + 1)
                return True

        return False


class ModelCheckpoint(Callback):
    """Save model checkpoints during training."""

    def __init__(self, filepath: str, monitor: str = 'val_loss',
                 save_best_only: bool = True, mode: str = 'min',
                 save_freq: int = 1, verbose: bool = True) -> None:
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_freq = save_freq
        self.verbose = verbose
        self.best_value: float = np.inf if mode == 'min' else -np.inf

    def on_train_begin(self) -> None:
        self.best_value = np.inf if self.mode == 'min' else -np.inf
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float],
                     model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer) -> bool:
        if (epoch + 1) % self.save_freq != 0:
            return False

        current_value = metrics.get(self.monitor)
        if current_value is None:
            if self.verbose:
                logger.warning("ModelCheckpoint: metric '%s' not found", self.monitor)
            return False

        should_save = not self.save_best_only

        if self.save_best_only:
            if self.mode == 'min':
                improved = current_value < self.best_value
            else:
                improved = current_value > self.best_value
            if improved:
                self.best_value = current_value
                should_save = True

        if should_save:
            filepath = self.filepath.format(epoch=epoch + 1)
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            model.save_weights(filepath)
            if self.verbose:
                logger.info("Saved checkpoint to %s (%s=%.6f)", filepath, self.monitor, current_value)

        return False


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when metric stops improving."""

    def __init__(self, monitor: str = 'val_loss', factor: float = 0.5,
                 patience: int = 5, min_lr: float = 1e-7,
                 mode: str = 'min', verbose: bool = True) -> None:
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.verbose = verbose
        self.wait: int = 0
        self.best_value: float = np.inf if mode == 'min' else -np.inf

    def on_train_begin(self) -> None:
        self.wait = 0
        self.best_value = np.inf if self.mode == 'min' else -np.inf

    def on_epoch_end(self, epoch: int, metrics: dict[str, float],
                     model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer) -> bool:
        current_value = metrics.get(self.monitor)
        if current_value is None:
            return False

        if self.mode == 'min':
            improved = current_value < self.best_value
        else:
            improved = current_value > self.best_value

        if improved:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(optimizer.learning_rate)
                new_lr = max(old_lr * self.factor, self.min_lr)
                if new_lr != old_lr:
                    optimizer.learning_rate.assign(new_lr)
                    if self.verbose:
                        logger.info("ReduceLROnPlateau: LR %.6f -> %.6f", old_lr, new_lr)
                    self.wait = 0

        return False


class TensorBoardLogger(Callback):
    """Log metrics to TensorBoard."""

    def __init__(self, log_dir: str, verbose: bool = True) -> None:
        self.log_dir = log_dir
        self.verbose = verbose
        self.writer: Optional[tf.summary.SummaryWriter] = None

    def on_train_begin(self) -> None:
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = tf.summary.create_file_writer(self.log_dir)
        if self.verbose:
            logger.info("TensorBoard logging to %s", self.log_dir)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float],
                     model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer) -> bool:
        if self.writer is None:
            return False

        with self.writer.as_default():
            for metric_name, metric_value in metrics.items():
                tf.summary.scalar(metric_name, metric_value, step=epoch)
            tf.summary.scalar('learning_rate', float(optimizer.learning_rate), step=epoch)

        self.writer.flush()
        return False

    def on_train_end(self) -> None:
        if self.writer:
            self.writer.close()


class LearningRateScheduler(Callback):
    """Custom learning rate scheduler."""

    def __init__(self, schedule_fn: Callable[[int], float], verbose: bool = True) -> None:
        self.schedule_fn = schedule_fn
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, metrics: dict[str, float],
                     model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer) -> bool:
        new_lr = self.schedule_fn(epoch)
        old_lr = float(optimizer.learning_rate)
        if new_lr != old_lr:
            optimizer.learning_rate.assign(new_lr)
            if self.verbose:
                logger.info("LR schedule: %.6f -> %.6f", old_lr, new_lr)
        return False


class TimerCallback(Callback):
    """Track training time."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.start_time: Optional[float] = None
        self.total_time: float = 0

    def on_train_begin(self) -> None:
        self.start_time = time.time()

    def on_epoch_end(self, epoch: int, metrics: dict[str, float],
                     model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer) -> bool:
        if self.start_time is not None:
            self.total_time = time.time() - self.start_time
        return False

    def on_train_end(self) -> None:
        if self.verbose:
            hours = int(self.total_time // 3600)
            minutes = int((self.total_time % 3600) // 60)
            seconds = int(self.total_time % 60)
            logger.info("Total training time: %dh %dm %ds", hours, minutes, seconds)
