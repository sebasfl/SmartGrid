from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import ConfigurationError


@dataclass
class DataConfig:
    """Data processing configuration."""

    # os.environ.get() always returns str here due to fallback defaults
    data_root: str = field(default_factory=lambda: os.environ.get("SMARTGRID_DATA_ROOT", "/app/data"))
    parquet_path: str = field(
        default_factory=lambda: os.environ.get(
            "SMARTGRID_PARQUET_PATH", "/app/data/processed/bdg2_electricity_cleaned.parquet"
        )
    )
    model_dir: str = field(default_factory=lambda: os.environ.get("SMARTGRID_MODEL_DIR", "/app/models"))

    # With 3h granularity: 60 days = 480 intervals, 30 days = 240 intervals
    forecast_horizon: int = 240
    lookback_window: int = 480
    stride: int = 8

    time_features: list[str] = field(
        default_factory=lambda: [
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
            "is_working_hours",
            "quarter",
            "day_of_year",
        ]
    )
    value_col: str = "value"

    scaler_sample_buildings: int = 10

    @property
    def input_dim(self) -> int:
        """Number of input features: time features + value column."""
        return len(self.time_features) + 1

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class CNNConfig:
    """CNN feature extractor configuration."""

    filters: list[int] = field(default_factory=lambda: [64, 128, 128])
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 3, 3])
    dropout: float = 0.2
    use_batch_norm: bool = True
    activation: str = "relu"


@dataclass
class LSTMConfig:
    """LSTM temporal encoder configuration."""

    units: list[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.2
    recurrent_dropout: float = 0.0
    use_bidirectional: bool = True


@dataclass
class ForecastHeadConfig:
    """Forecasting head configuration."""

    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.2
    activation: str = "relu"


@dataclass
class LossConfig:
    """Loss function configuration."""

    forecast_loss_type: str = "mse"
    huber_delta: float = 1.0


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    optimizer_type: str = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-7

    use_lr_schedule: bool = True
    lr_schedule_type: str = "cosine"
    warmup_steps: int = 1000
    min_lr: float = 1e-7

    gradient_clip_norm: float = 1.0


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    batch_size: int = 8
    epochs: int = 50
    validation_freq: int = 1

    use_mixed_precision: bool = True
    gpu_id: int = 0

    save_checkpoint_freq: int = 5
    keep_last_n_checkpoints: int = 3

    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    early_stopping_monitor: str = "val_loss"

    log_freq: int = 100
    tensorboard_dir: str = "/app/logs"

    random_seed: int = 42


_KNOWN_SECTIONS = frozenset({"data", "cnn", "lstm", "forecast_head", "loss", "optimizer", "training"})

_SECTION_CLS = {
    "data": DataConfig,
    "cnn": CNNConfig,
    "lstm": LSTMConfig,
    "forecast_head": ForecastHeadConfig,
    "loss": LossConfig,
    "optimizer": OptimizerConfig,
    "training": TrainingConfig,
}


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
    def from_json(cls, json_path: str) -> Config:
        """Load configuration from JSON file, validating all keys."""
        with open(json_path) as f:
            config_dict: dict[str, dict] = json.load(f)

        unknown_sections = set(config_dict.keys()) - _KNOWN_SECTIONS
        if unknown_sections:
            raise ConfigurationError(
                f"Unknown config sections: {unknown_sections}. Valid sections: {sorted(_KNOWN_SECTIONS)}"
            )

        kwargs: dict = {}
        for section, section_cls in _SECTION_CLS.items():
            section_data = config_dict.get(section, {})
            valid_fields = {f.name for f in section_cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            unknown_keys = set(section_data.keys()) - valid_fields
            if unknown_keys:
                raise ConfigurationError(
                    f"Unknown keys in '{section}': {unknown_keys}. Valid keys: {sorted(valid_fields)}"
                )
            kwargs[section] = section_cls(**section_data)

        return cls(**kwargs)

    def to_json(self, json_path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = {name: getattr(self, name).__dict__ for name in _SECTION_CLS}

        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def validate(self) -> None:
        """Validate configuration consistency. Raises ConfigurationError on failure."""
        if len(self.cnn.filters) != len(self.cnn.kernel_sizes):
            raise ConfigurationError(
                f"CNN filters ({len(self.cnn.filters)}) and kernel_sizes "
                f"({len(self.cnn.kernel_sizes)}) must have same length"
            )
        if self.data.lookback_window <= 0:
            raise ConfigurationError("lookback_window must be positive")
        if self.data.forecast_horizon <= 0:
            raise ConfigurationError("forecast_horizon must be positive")
        ratios = self.data.train_ratio + self.data.val_ratio + self.data.test_ratio
        if abs(ratios - 1.0) > 0.01:
            raise ConfigurationError(f"Data split ratios must sum to 1.0, got {ratios:.3f}")
        if self.training.batch_size <= 0:
            raise ConfigurationError("batch_size must be positive")
        if self.optimizer.learning_rate <= 0:
            raise ConfigurationError("learning_rate must be positive")

    def __str__(self) -> str:
        lines = ["=" * 60, "CONFIGURATION", "=" * 60, ""]

        for section_name, attr_name in [
            ("DATA", "data"),
            ("CNN", "cnn"),
            ("LSTM", "lstm"),
            ("FORECAST HEAD", "forecast_head"),
            ("LOSS", "loss"),
            ("OPTIMIZER", "optimizer"),
            ("TRAINING", "training"),
        ]:
            section = getattr(self, attr_name)
            lines.append(f"{section_name}:")
            for key, value in section.__dict__.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


default_config = Config()
