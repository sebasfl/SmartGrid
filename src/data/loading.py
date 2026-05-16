"""Shared data loading utilities for training and inference."""
from __future__ import annotations

import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_parquet(path: str) -> pd.DataFrame:
    """Load a preprocessed parquet file and log basic stats."""
    logger.info("Loading data from %s...", path)
    df = pd.read_parquet(path)
    logger.info(
        "Loaded %s records for %d buildings",
        f"{len(df):,}",
        df["building_id"].nunique(),
    )
    return df


def load_building_split(split_path: str) -> dict[str, list[str]]:
    """Load a building split JSON file.

    Returns:
        Dict with keys 'train', 'validation', 'test' mapping to building ID lists.
    """
    with open(split_path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def filter_by_building_ids(
    df: pd.DataFrame,
    building_ids: list[str],
) -> pd.DataFrame:
    """Filter dataframe to include only the given building IDs."""
    filtered = df[df["building_id"].isin(building_ids)]
    logger.info("  Filtered to %s records", f"{len(filtered):,}")
    return filtered
