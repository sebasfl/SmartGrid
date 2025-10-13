# src/training/callbacks.py
# Training callbacks for MTL model
import tensorflow as tf
from pathlib import Path
import numpy as np
import time
from typing import Optional


class Callback:
    """Base callback class."""

    def on_train_begin(self):
        """Called at the start of training."""
        pass

    def on_epoch_end(self, epoch: int, metrics: dict, model, optimizer) -> bool:
        """Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            metrics: Dict of metrics from training/validation
            model: The MTL model
            optimizer: The optimizer

        Returns:
            Boolean indicating whether to stop training
        """
        return False

    def on_train_end(self):
        """Called at the end of training."""
        pass


class EarlyStopping(Callback):
    """Stop training when monitored metric stops improving."""

    def __init__(self, monitor: str = 'val_loss', patience: int = 10,
                 min_delta: float = 1e-4, mode: str = 'min',
                 verbose: bool = True):
        """
        Args:
            monitor: Metric to monitor (e.g., 'val_loss', 'loss')
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
            verbose: Print messages
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.wait = 0
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.stopped_epoch = 0

    def on_train_begin(self):
        """Reset state at training start."""
        self.wait = 0
        self.best_value = np.inf if self.mode == 'min' else -np.inf
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, metrics, model, optimizer):
        """Check if training should stop."""
        current_value = metrics.get(self.monitor)

        if current_value is None:
            if self.verbose:
                print(f"⚠️  EarlyStopping: metric '{self.monitor}' not found in metrics")
            return False

        # Check for improvement
        if self.mode == 'min':
            improved = (self.best_value - current_value) > self.min_delta
        else:
            improved = (current_value - self.best_value) > self.min_delta

        if improved:
            self.best_value = current_value
            self.wait = 0
            if self.verbose:
                print(f"🎯 EarlyStopping: {self.monitor} improved to {current_value:.6f}")
        else:
            self.wait += 1
            if self.verbose:
                print(f"⏳ EarlyStopping: {self.monitor} did not improve ({self.wait}/{self.patience})")

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f"🛑 EarlyStopping: stopping training at epoch {epoch + 1}")
                return True  # Stop training

        return False


class ModelCheckpoint(Callback):
    """Save model checkpoints during training."""

    def __init__(self, filepath: str, monitor: str = 'val_loss',
                 save_best_only: bool = True, mode: str = 'min',
                 save_freq: int = 1, verbose: bool = True):
        """
        Args:
            filepath: Path to save model (can include {epoch} placeholder)
            monitor: Metric to monitor
            save_best_only: Only save when metric improves
            mode: 'min' or 'max'
            save_freq: Save every N epochs
            verbose: Print messages
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_freq = save_freq
        self.verbose = verbose

        self.best_value = np.inf if mode == 'min' else -np.inf

    def on_train_begin(self):
        """Initialize best value."""
        self.best_value = np.inf if self.mode == 'min' else -np.inf
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, metrics, model, optimizer):
        """Save checkpoint if conditions are met."""
        if (epoch + 1) % self.save_freq != 0:
            return False

        current_value = metrics.get(self.monitor)

        if current_value is None:
            if self.verbose:
                print(f"⚠️  ModelCheckpoint: metric '{self.monitor}' not found")
            return False

        # Check if we should save
        should_save = False

        if self.save_best_only:
            if self.mode == 'min':
                improved = current_value < self.best_value
            else:
                improved = current_value > self.best_value

            if improved:
                self.best_value = current_value
                should_save = True
        else:
            should_save = True

        if should_save:
            # Format filepath with epoch number
            filepath = self.filepath.format(epoch=epoch + 1)
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Save model
            model.save_weights(filepath)

            if self.verbose:
                print(f"💾 Saved model checkpoint to {filepath} ({self.monitor}={current_value:.6f})")

        return False


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when metric stops improving."""

    def __init__(self, monitor: str = 'val_loss', factor: float = 0.5,
                 patience: int = 5, min_lr: float = 1e-7,
                 mode: str = 'min', verbose: bool = True):
        """
        Args:
            monitor: Metric to monitor
            factor: Factor to reduce LR by (new_lr = lr * factor)
            patience: Number of epochs with no improvement to wait
            min_lr: Minimum learning rate
            mode: 'min' or 'max'
            verbose: Print messages
        """
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.verbose = verbose

        self.wait = 0
        self.best_value = np.inf if mode == 'min' else -np.inf

    def on_train_begin(self):
        """Reset state."""
        self.wait = 0
        self.best_value = np.inf if self.mode == 'min' else -np.inf

    def on_epoch_end(self, epoch, metrics, model, optimizer):
        """Reduce LR if metric plateaus."""
        current_value = metrics.get(self.monitor)

        if current_value is None:
            return False

        # Check for improvement
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
                # Reduce learning rate
                old_lr = float(optimizer.learning_rate)
                new_lr = max(old_lr * self.factor, self.min_lr)

                if new_lr != old_lr:
                    optimizer.learning_rate.assign(new_lr)
                    if self.verbose:
                        print(f"📉 ReduceLROnPlateau: reducing LR from {old_lr:.6f} to {new_lr:.6f}")
                    self.wait = 0

        return False


class TensorBoardLogger(Callback):
    """Log metrics to TensorBoard."""

    def __init__(self, log_dir: str, verbose: bool = True):
        """
        Args:
            log_dir: Directory to save TensorBoard logs
            verbose: Print messages
        """
        self.log_dir = log_dir
        self.verbose = verbose
        self.writer = None

    def on_train_begin(self):
        """Create TensorBoard writer."""
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.writer = tf.summary.create_file_writer(self.log_dir)

        if self.verbose:
            print(f"📊 TensorBoard logging to {self.log_dir}")

    def on_epoch_end(self, epoch, metrics, model, optimizer):
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return False

        with self.writer.as_default():
            for metric_name, metric_value in metrics.items():
                tf.summary.scalar(metric_name, metric_value, step=epoch)

            # Log learning rate
            lr = float(optimizer.learning_rate)
            tf.summary.scalar('learning_rate', lr, step=epoch)

        self.writer.flush()
        return False

    def on_train_end(self):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()


class LearningRateScheduler(Callback):
    """Custom learning rate scheduler."""

    def __init__(self, schedule_fn, verbose: bool = True):
        """
        Args:
            schedule_fn: Function that takes epoch and returns new LR
            verbose: Print messages
        """
        self.schedule_fn = schedule_fn
        self.verbose = verbose

    def on_epoch_end(self, epoch, metrics, model, optimizer):
        """Update learning rate based on schedule."""
        new_lr = self.schedule_fn(epoch)
        old_lr = float(optimizer.learning_rate)

        if new_lr != old_lr:
            optimizer.learning_rate.assign(new_lr)
            if self.verbose:
                print(f"📊 LR schedule: {old_lr:.6f} → {new_lr:.6f}")

        return False


class TimerCallback(Callback):
    """Track training time."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.start_time = None
        self.total_time = 0

    def on_train_begin(self):
        """Start timer."""
        self.start_time = time.time()

    def on_epoch_end(self, epoch, metrics, model, optimizer):
        """Track elapsed time."""
        self.total_time = time.time() - self.start_time
        return False

    def on_train_end(self):
        """Print total training time."""
        if self.verbose:
            hours = int(self.total_time // 3600)
            minutes = int((self.total_time % 3600) // 60)
            seconds = int(self.total_time % 60)
            print(f"⏱️  Total training time: {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    print("Testing callbacks...")

    # Mock metrics for testing
    test_metrics = {
        'loss': 0.5,
        'val_loss': 0.6,
        'anomaly_loss': 0.2,
        'forecast_loss': 0.3
    }

    # Test EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    early_stop.on_train_begin()

    for epoch in range(10):
        # Simulate decreasing loss, then plateau
        test_metrics['val_loss'] = 0.6 - epoch * 0.05 if epoch < 5 else 0.35

        should_stop = early_stop.on_epoch_end(epoch, test_metrics, None, None)
        if should_stop:
            print(f"Training would stop at epoch {epoch}")
            break

    print("✅ Callbacks work correctly!")
