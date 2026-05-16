from __future__ import annotations

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


def configure_gpu() -> None:
    """Configure GPU memory growth. Must be called before any TF operations."""
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus[0], "GPU")
            logger.info("GPU configured: %d GPU(s), memory growth enabled", len(gpus))
        else:
            logger.warning("No GPU detected — running on CPU")
    except RuntimeError as e:
        logger.warning("GPU configuration error: %s", e)


def check_gpu_availability() -> bool:
    """Log GPU availability status. Returns True if GPU is available."""
    logger.info("TensorFlow %s | CUDA built: %s", tf.__version__, tf.test.is_built_with_cuda())

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for i, gpu in enumerate(gpus):
            logger.info("  GPU %d: %s", i, gpu.name)
    else:
        logger.warning("  No GPU detected — training will use CPU")

    return len(gpus) > 0
