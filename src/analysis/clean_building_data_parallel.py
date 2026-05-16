from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

os.environ['NUMBA_CUDA_ENABLE_PYNVJITLINK'] = '1'

try:
    import cudf
    import cupy as cp  # noqa: F401
    GPU_AVAILABLE = True
    logger.info("GPU acceleration: cuDF + CuPy available")
except (ImportError, ModuleNotFoundError):
    GPU_AVAILABLE = False

DataFrame = Union[pd.DataFrame, "cudf.DataFrame"]


class BasicPreprocessor:
    """Preprocessing pipeline: time features, optional 3h resampling, deduplication."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.gpu_available = GPU_AVAILABLE

    def load_data(self, parquet_path: Path) -> DataFrame:
        logger.info("Loading data %s...", "on GPU" if self.gpu_available else "on CPU")

        if self.gpu_available:
            df = cudf.read_parquet(parquet_path)
        else:
            df = pd.read_parquet(parquet_path)

        ts_col = "timestamp_local" if "timestamp_local" in df.columns else "timestamp_utc"

        if "meter" in df.columns:
            df = df[df["meter"] == "electricity"]

        if self.gpu_available:
            df["building_id"] = df["building_id"].astype("string")
            if "meter" in df.columns:
                df["meter"] = df["meter"].astype("string")
            df["timestamp_local"] = cudf.to_datetime(df["timestamp_local"])
        else:
            df["building_id"] = df["building_id"].astype(str)
            if "meter" in df.columns:
                df["meter"] = df["meter"].astype(str)
            df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])

        df["value"] = df["value"].astype("float32")

        if ts_col != "timestamp_local":
            df = df.rename(columns={ts_col: "timestamp_local"})

        df = df.sort_values(["building_id", "timestamp_local"]).reset_index(drop=True)
        return df

    def add_time_features(self, df: DataFrame) -> DataFrame:
        if self.verbose:
            logger.info("Adding time features (vectorized)...")

        if not self.gpu_available:
            df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])

        df['hour'] = df['timestamp_local'].dt.hour.astype('int8')
        df['day_of_week'] = df['timestamp_local'].dt.dayofweek.astype('int8')
        df['month'] = df['timestamp_local'].dt.month.astype('int8')
        df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
        df['is_working_hours'] = (
            (df['hour'] >= 8) & (df['hour'] <= 18) & (df['is_weekend'] == 0)
        ).astype('int8')
        df['quarter'] = df['timestamp_local'].dt.quarter.astype('int8')
        df['day_of_year'] = df['timestamp_local'].dt.dayofyear.astype('int16')

        return df

    def resample_to_3h(self, df: DataFrame) -> DataFrame:
        if self.verbose:
            logger.info("Resampling from 1h to 3h intervals...")

        before_resample = len(df)

        if self.gpu_available:
            df['time_bin'] = (
                df['timestamp_local'].astype('int64') // (3 * 3600 * 1_000_000_000)
            ) * (3 * 3600 * 1_000_000_000)
            df['time_bin'] = df['time_bin'].astype('datetime64[ns]')
        else:
            df['time_bin'] = df['timestamp_local'].dt.floor('3h')

        agg_dict: dict = {'value': 'mean'}
        if 'meter' in df.columns:
            agg_dict['meter'] = 'first'

        df = df.groupby(['building_id', 'time_bin'], as_index=False).agg(agg_dict)
        df = df.rename(columns={'time_bin': 'timestamp_local'})
        df = df.dropna(subset=['value'])

        if self.verbose:
            logger.info(
                "  %s -> %s records (%.1f%% reduction)",
                f"{before_resample:,}", f"{len(df):,}",
                (1 - len(df) / before_resample) * 100,
            )

        return df

    def remove_duplicates(self, df: DataFrame) -> DataFrame:
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['building_id', 'timestamp_local'], keep='first')

        if self.verbose:
            removed = before_dedup - len(df)
            if removed > 0:
                logger.info("  Removed %s duplicate records", f"{removed:,}")

        return df

    def process(
        self,
        df: DataFrame,
        skip_deduplication: bool = False,
        resample_3h: bool = False,
    ) -> DataFrame:
        if self.verbose:
            logger.info("Starting preprocessing: %s records, %s buildings (%s)",
                        f"{len(df):,}", f"{df['building_id'].nunique():,}",
                        "GPU" if self.gpu_available else "CPU")

        original_size = len(df)

        df = self.add_time_features(df)

        if resample_3h:
            df = self.resample_to_3h(df)
            df = self.add_time_features(df)

        if not skip_deduplication:
            df = self.remove_duplicates(df)

        if self.verbose:
            logger.info(
                "Preprocessing complete: %.1f%% retained (%s -> %s records)",
                (len(df) / original_size) * 100,
                f"{original_size:,}", f"{len(df):,}",
            )

        return df


def _build_arrow_schema(columns: list[str]) -> pa.Schema:
    """Build Arrow schema matching the preprocessed DataFrame columns."""
    schema_fields = [
        ('timestamp_local', pa.timestamp('ns')),
        ('building_id', pa.string()),
        ('value', pa.float32()),
    ]

    if 'meter' in columns:
        schema_fields.insert(2, ('meter', pa.string()))

    from ..config import DataConfig
    time_cols = DataConfig().time_features
    for tc in time_cols:
        if tc in columns:
            schema_fields.append((tc, pa.bool_() if tc.startswith('is_') else pa.int32()))

    return pa.schema(schema_fields)


def main() -> None:
    parser = argparse.ArgumentParser(description='Preprocessing: time features + deduplication')
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-deduplicate', action='store_true')
    parser.add_argument('--resample-3h', action='store_true',
                        help='Resample from 1h to 3h intervals (reduces memory ~66%%)')

    args = parser.parse_args()

    processor = BasicPreprocessor(verbose=True)
    df = processor.load_data(args.parquet)
    df_clean = processor.process(df, skip_deduplication=args.no_deduplicate, resample_3h=args.resample_3h)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if processor.gpu_available:
        df_pandas = df_clean.to_pandas()
    else:
        df_pandas = df_clean

    schema = _build_arrow_schema(df_pandas.columns.tolist())
    table = pa.Table.from_pandas(df_pandas, schema=schema, preserve_index=False)
    pq.write_table(table, args.output, compression='snappy')

    logger.info("Saved %s records to %s", f"{len(df_clean):,}", args.output)


if __name__ == "__main__":
    main()
