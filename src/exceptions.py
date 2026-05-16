"""Custom exceptions for the SmartGrid pipeline."""

from __future__ import annotations


class SmartGridError(Exception):
    """Base exception for all SmartGrid errors."""


class ConfigurationError(SmartGridError):
    """Invalid or inconsistent configuration."""


class DataError(SmartGridError):
    """Data loading, validation, or preprocessing failure."""


class InsufficientDataError(DataError):
    """Not enough data to create sequences or train a model."""


class DataQualityError(DataError):
    """Data contains too many NaN/Inf values or is otherwise unusable."""


class ModelBuildError(SmartGridError):
    """Model construction or compilation failure."""


class TrainingError(SmartGridError):
    """Error during model training."""
