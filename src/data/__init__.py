from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "create_sequences":
        from .sequences import create_sequences

        return create_sequences
    if name == "prepare_dataset":
        from .dataset import prepare_dataset

        return prepare_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["create_sequences", "prepare_dataset"]
