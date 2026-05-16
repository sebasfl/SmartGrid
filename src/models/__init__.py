from __future__ import annotations

from typing import Any

_EXPORTS = {
    "CNNFeatureExtractor",
    "LSTMTemporalEncoder",
    "ForecastingHead",
    "HybridCNNLSTM",
    "build_cnn_lstm_model",
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        from . import cnn_lstm

        return getattr(cnn_lstm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_EXPORTS)
