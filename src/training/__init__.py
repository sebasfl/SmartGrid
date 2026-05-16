from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "CNNLSTMTrainer":
        from .trainer import CNNLSTMTrainer

        return CNNLSTMTrainer
    if name in ("EarlyStopping", "ModelCheckpoint", "ReduceLROnPlateau", "TensorBoardLogger", "TimerCallback"):
        from . import callbacks

        return getattr(callbacks, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CNNLSTMTrainer",
    "EarlyStopping",
    "ModelCheckpoint",
    "ReduceLROnPlateau",
    "TensorBoardLogger",
    "TimerCallback",
]
