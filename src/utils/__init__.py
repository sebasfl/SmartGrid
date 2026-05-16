from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name in ("configure_gpu", "check_gpu_availability"):
        from .gpu import check_gpu_availability, configure_gpu  # noqa: F401

        return locals()[name]
    if name == "set_random_seed":
        from .reproducibility import set_random_seed

        return set_random_seed
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["configure_gpu", "check_gpu_availability", "set_random_seed"]
