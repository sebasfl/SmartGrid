from __future__ import annotations

import os

import numpy as np
import tensorflow as tf


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all frameworks."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    np.random.seed(seed)
    tf.random.set_seed(seed)
