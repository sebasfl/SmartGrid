# src/config.py
# Centralized configuration for Hybrid CNN-LSTM model
from dataclasses import dataclass, field
from typing import List
import json
from pathlib import Path


@dataclass
class DataConfig:
    """Data processing configuration."""
    # Paths
    data_root: str = "/app/data"
    parquet_path: str = "/app/data/processed/bdg2_electricity_cleaned.parquet"
    model_dir: str = "/app/models"

    # Sequence generation
    # Note: With 3-hour granularity, divide original hourly values by 3
    # Use 2x lookback data relative to forecast horizon (2 months to predict 1 month)
    forecast_horizon: int = 240  # 1 month ahead (30 days * 8 intervals/day with 3h granularity)
    lookback_window: int = 480  # 2 months lookback (60 days * 8 intervals/day with 3h granularity)
    stride: int = 8  # Sliding window stride (1 day with 3h intervals = 8 intervals)

    # Features
    time_features: List[str] = field(default_factory=lambda: [
        'hour', 'day_of_week', 'month', 'is_weekend',
        'is_working_hours', 'quarter', 'day_of_year'
    ])
    value_col: str = 'value'

    # Splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class CNNConfig:
    """CNN feature extractor configuration."""
    # Convolutional layers
    filters: List[int] = field(default_factory=lambda: [64, 128, 128])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])

    # Regularization
    dropout: float = 0.2
    use_batch_norm: bool = True

    # Activation
    activation: str = 'relu'  # 'relu', 'gelu', 'swish'


@dataclass
class LSTMConfig:
    """LSTM temporal encoder configuration."""
    # LSTM layers
    units: List[int] = field(default_factory=lambda: [128, 64])

    # Regularization
    dropout: float = 0.2
    recurrent_dropout: float = 0.0  # 0.0 enables cuDNN acceleration (3-5x faster)

    # Architecture
    use_bidirectional: bool = True


@dataclass
class ForecastHeadConfig:
    """Forecasting head configuration."""
    # Dense layers
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])

    # Regularization
    dropout: float = 0.2

    # Activation
    activation: str = 'relu'


@dataclass
class LossConfig:
    """Loss function configuration."""
    # Forecast loss
    forecast_loss_type: str = 'mse'  # 'mse', 'mae', 'huber'
    huber_delta: float = 1.0


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    optimizer_type: str = 'adam'  # 'adam', 'adamw', 'sgd'
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-7

    # Learning rate schedule
    use_lr_schedule: bool = True
    lr_schedule_type: str = 'cosine'  # 'cosine', 'step', 'exponential'
    warmup_steps: int = 1000
    min_lr: float = 1e-7

    # Gradient clipping
    gradient_clip_norm: float = 1.0


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    # Basic training
    batch_size: int = 8  # Reduced due to longer sequences
    epochs: int = 50
    validation_freq: int = 1  # Validate every N epochs

    # GPU settings
    use_mixed_precision: bool = True  # FP16 training
    gpu_id: int = 0

    # Checkpointing
    save_checkpoint_freq: int = 5  # Save every N epochs
    keep_last_n_checkpoints: int = 3

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    early_stopping_monitor: str = 'val_loss'  # 'val_loss', 'val_mae', etc.

    # Logging
    log_freq: int = 100  # Log every N batches
    tensorboard_dir: str = "/app/logs"

    # Random seed
    random_seed: int = 42


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    forecast_head: ForecastHeadConfig = field(default_factory=ForecastHeadConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_json(cls, json_path: str):
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)

        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            cnn=CNNConfig(**config_dict.get('cnn', {})),
            lstm=LSTMConfig(**config_dict.get('lstm', {})),
            forecast_head=ForecastHeadConfig(**config_dict.get('forecast_head', {})),
            loss=LossConfig(**config_dict.get('loss', {})),
            optimizer=OptimizerConfig(**config_dict.get('optimizer', {})),
            training=TrainingConfig(**config_dict.get('training', {}))
        )

    def to_json(self, json_path: str):
        """Save configuration to JSON file."""
        config_dict = {
            'data': self.data.__dict__,
            'cnn': self.cnn.__dict__,
            'lstm': self.lstm.__dict__,
            'forecast_head': self.forecast_head.__dict__,
            'loss': self.loss.__dict__,
            'optimizer': self.optimizer.__dict__,
            'training': self.training.__dict__
        }

        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def __str__(self):
        """Pretty print configuration."""
        lines = ["=" * 60, "CONFIGURATION", "=" * 60, ""]

        for section_name, section in [
            ("DATA", self.data),
            ("CNN", self.cnn),
            ("LSTM", self.lstm),
            ("FORECAST HEAD", self.forecast_head),
            ("LOSS", self.loss),
            ("OPTIMIZER", self.optimizer),
            ("TRAINING", self.training)
        ]:
            lines.append(f"{section_name}:")
            for key, value in section.__dict__.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


# Default configuration instance
default_config = Config()


if __name__ == "__main__":
    # Example: Save and load config
    config = Config()

    # Customize if needed
    config.cnn.filters = [128, 256, 256]
    config.training.batch_size = 16

    # Save to JSON
    config.to_json("config_example.json")
    print("Config saved to config_example.json")

    # Load from JSON
    loaded_config = Config.from_json("config_example.json")
    print("\nLoaded configuration:")
    print(loaded_config)
