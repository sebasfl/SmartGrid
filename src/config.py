# src/config.py
# Centralized configuration for MTL Transformer
from dataclasses import dataclass, field
from typing import List, Tuple
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
    lookback_window: int = 168  # 7 days in hours
    forecast_horizon: int = 24  # 1 day ahead
    stride: int = 1  # Sliding window stride

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

    # Anomaly labeling
    anomaly_method: str = 'iqr'  # 'iqr' or 'zscore'
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0


@dataclass
class TransformerConfig:
    """Transformer encoder configuration."""
    d_model: int = 128  # Embedding dimension
    num_heads: int = 8  # Multi-head attention heads
    num_layers: int = 4  # Encoder layers
    ff_dim: int = 512  # Feed-forward hidden dimension
    dropout: float = 0.1
    activation: str = 'gelu'  # 'relu', 'gelu', 'swish'
    use_positional_encoding: bool = True
    positional_encoding_type: str = 'sinusoidal'  # 'sinusoidal' or 'learnable'
    max_seq_length: int = 512


@dataclass
class MTLHeadsConfig:
    """Multi-task learning heads configuration."""
    # Anomaly detection head
    anomaly_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    anomaly_dropout: float = 0.3
    anomaly_activation: str = 'relu'

    # Forecasting head
    forecast_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    forecast_dropout: float = 0.2
    forecast_activation: str = 'relu'
    use_decoder: bool = False  # Use transformer decoder for forecasting


@dataclass
class LossConfig:
    """Loss function configuration."""
    # Multi-task loss weights
    alpha_anomaly: float = 0.3  # Weight for anomaly detection loss
    beta_forecast: float = 0.7  # Weight for forecasting loss

    # Anomaly loss
    anomaly_loss_type: str = 'bce'  # 'bce' or 'focal'
    focal_alpha: float = 0.25  # For focal loss
    focal_gamma: float = 2.0

    # Forecast loss
    forecast_loss_type: str = 'mse'  # 'mse', 'mae', 'huber'
    huber_delta: float = 1.0

    # Uncertainty weighting (Kendall et al. 2018)
    use_uncertainty_weighting: bool = False


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
    batch_size: int = 32
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
    early_stopping_monitor: str = 'val_loss'  # 'val_loss', 'val_forecast_rmse', etc.

    # Logging
    log_freq: int = 100  # Log every N batches
    tensorboard_dir: str = "/app/logs"

    # Random seed
    random_seed: int = 42


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    mtl_heads: MTLHeadsConfig = field(default_factory=MTLHeadsConfig)
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
            transformer=TransformerConfig(**config_dict.get('transformer', {})),
            mtl_heads=MTLHeadsConfig(**config_dict.get('mtl_heads', {})),
            loss=LossConfig(**config_dict.get('loss', {})),
            optimizer=OptimizerConfig(**config_dict.get('optimizer', {})),
            training=TrainingConfig(**config_dict.get('training', {}))
        )

    def to_json(self, json_path: str):
        """Save configuration to JSON file."""
        config_dict = {
            'data': self.data.__dict__,
            'transformer': self.transformer.__dict__,
            'mtl_heads': self.mtl_heads.__dict__,
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
            ("TRANSFORMER", self.transformer),
            ("MTL HEADS", self.mtl_heads),
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
    config.transformer.d_model = 256
    config.training.batch_size = 64

    # Save to JSON
    config.to_json("config_example.json")
    print("Config saved to config_example.json")

    # Load from JSON
    loaded_config = Config.from_json("config_example.json")
    print("\nLoaded configuration:")
    print(loaded_config)
