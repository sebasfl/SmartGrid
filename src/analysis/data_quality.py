from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cudf = None
    cp = np

logger = logging.getLogger(__name__)


def compute_quality_score(
    completeness_ratio: float,
    coverage_ratio: float,
    continuity_ratio: float,
    zero_ratio: float,
    constant_ratio: float,
    duplicate_ratio: float,
) -> float:
    """Weighted quality score formula, shared between CPU and GPU paths."""
    return (
        0.30 * completeness_ratio
        + 0.25 * coverage_ratio
        + 0.25 * continuity_ratio
        + 0.10 * (1.0 - zero_ratio)
        + 0.05 * (1.0 - constant_ratio)
        + 0.05 * (1.0 - duplicate_ratio)
    )


def calculate_building_quality_metrics(
    df: pd.DataFrame,
    building_id: str,
    granularity_hours: int = 1,
) -> dict | None:
    """Calculate data quality metrics for a single building (CPU path)."""
    building_df = df[df['building_id'] == building_id].copy()

    if len(building_df) == 0:
        return None

    building_df = building_df.sort_values('timestamp_local')
    total_rows = len(building_df)

    missing_values = int(building_df['value'].isna().sum())
    completeness_ratio = 1.0 - (missing_values / total_rows) if total_rows > 0 else 0.0

    if len(building_df) > 1:
        time_span_hours = (
            building_df['timestamp_local'].max() - building_df['timestamp_local'].min()
        ).total_seconds() / 3600
        expected_records = int(time_span_hours / granularity_hours) + 1
        coverage_ratio = min(1.0, total_rows / expected_records) if expected_records > 0 else 0.0
    else:
        time_span_hours = 0.0
        coverage_ratio = 0.0

    if len(building_df) > 1:
        time_diff = building_df['timestamp_local'].diff()
        gap_threshold = pd.Timedelta(hours=2 * granularity_hours)
        gaps_2h = int((time_diff > gap_threshold).sum())
        gaps_24h = int((time_diff > pd.Timedelta(hours=24)).sum())
        continuity_ratio = 1.0 - (gaps_2h / (total_rows - 1)) if total_rows > 1 else 0.0
    else:
        gaps_2h = 0
        gaps_24h = 0
        continuity_ratio = 0.0

    values = building_df['value'].dropna()
    if len(values) > 0:
        zero_ratio = float((values == 0).sum() / len(values))

        constant_threshold = int(48 / granularity_hours)
        is_same = values == values.shift(1)
        groups = (~is_same).cumsum()
        run_lengths = is_same.groupby(groups).sum()
        constant_sequences = int((run_lengths >= constant_threshold).sum())
        constant_ratio = constant_sequences / len(values)

        mean_value = float(values.mean())
        std_value = float(values.std())
        median_value = float(values.median())
        cv = std_value / mean_value if mean_value > 0 else 0.0
        min_value = float(values.min())
        max_value = float(values.max())
    else:
        zero_ratio = 1.0
        constant_ratio = 1.0
        mean_value = std_value = median_value = cv = min_value = max_value = 0.0

    duplicate_timestamps = int(building_df['timestamp_local'].duplicated().sum())
    duplicate_ratio = duplicate_timestamps / total_rows if total_rows > 0 else 0.0

    quality_score = compute_quality_score(
        completeness_ratio, coverage_ratio, continuity_ratio,
        zero_ratio, constant_ratio, duplicate_ratio,
    )

    return {
        'building_id': building_id,
        'total_records': total_rows,
        'time_span_hours': time_span_hours,
        'time_span_days': time_span_hours / 24,
        'completeness_ratio': completeness_ratio,
        'coverage_ratio': coverage_ratio,
        'continuity_ratio': continuity_ratio,
        'gaps_2h': gaps_2h,
        'gaps_24h': gaps_24h,
        'zero_ratio': zero_ratio,
        'constant_ratio': constant_ratio,
        'duplicate_ratio': duplicate_ratio,
        'mean_value': mean_value,
        'std_value': std_value,
        'median_value': median_value,
        'min_value': min_value,
        'max_value': max_value,
        'coefficient_of_variation': cv,
        'quality_score': quality_score,
    }


def evaluate_all_buildings_gpu(gdf: cudf.DataFrame, granularity_hours: int = 1) -> pd.DataFrame:
    """GPU-accelerated vectorized quality evaluation for all buildings."""
    logger.info("GPU-accelerated evaluation (granularity: %dh)...", granularity_hours)

    gdf = gdf.sort_values(['building_id', 'timestamp_local'])

    building_stats = gdf.groupby('building_id').agg({
        'value': ['mean', 'std', 'median', 'min', 'max'],
        'timestamp_local': ['min', 'max', 'size'],
    }).reset_index()
    building_stats.columns = ['_'.join(col).strip('_') for col in building_stats.columns.values]
    building_stats = building_stats.rename(columns={'timestamp_local_size': 'value_count'})

    time_diff_seconds = (
        (building_stats['timestamp_local_max'] - building_stats['timestamp_local_min'])
        .astype('int64') / 1_000_000_000
    )
    building_stats['time_span_hours'] = time_diff_seconds / 3600.0
    building_stats['time_span_days'] = building_stats['time_span_hours'] / 24.0

    missing_counts = (
        gdf['value'].isna().groupby(gdf['building_id']).sum()
        .reset_index().rename(columns={'value': 'missing_count'})
    )
    missing_counts.columns = ['building_id', 'missing_count']
    building_stats = building_stats.merge(missing_counts, on='building_id')
    building_stats['completeness_ratio'] = 1.0 - (
        building_stats['missing_count'].astype('float64') / building_stats['value_count'].astype('float64')
    )

    building_stats['expected_records'] = (
        building_stats['time_span_hours'].astype('int64') / granularity_hours
    ).astype('int64') + 1
    building_stats['coverage_ratio'] = (
        building_stats['value_count'].astype('float64') / building_stats['expected_records'].astype('float64')
    ).clip(upper=1.0)

    zero_counts = (
        (gdf['value'] == 0).groupby(gdf['building_id']).sum()
        .reset_index().rename(columns={'value': 'zero_count'})
    )
    zero_counts.columns = ['building_id', 'zero_count']
    building_stats = building_stats.merge(zero_counts, on='building_id')
    building_stats['zero_ratio'] = (
        building_stats['zero_count'].astype('float64') / building_stats['value_count'].astype('float64')
    )

    gdf['time_diff'] = gdf.groupby('building_id')['timestamp_local'].diff()
    time_diff_ns = gdf['time_diff'].astype('int64')
    gap_threshold_ns = 2 * granularity_hours * 3600 * 1_000_000_000

    gdf['gap_2h'] = (time_diff_ns > gap_threshold_ns).astype('int32')
    gdf['gap_24h'] = (time_diff_ns > 24 * 3600 * 1_000_000_000).astype('int32')

    gap_stats = gdf.groupby('building_id').agg({'gap_2h': 'sum', 'gap_24h': 'sum'}).reset_index()
    gap_stats.columns = ['building_id', 'gaps_2h', 'gaps_24h']
    building_stats = building_stats.merge(gap_stats, on='building_id')

    building_stats['continuity_ratio'] = (
        1.0 - building_stats['gaps_2h'].astype('float64')
        / (building_stats['value_count'].astype('float64') - 1.0)
    ).fillna(0.0)

    gdf['is_dup'] = (
        gdf.sort_values(['building_id', 'timestamp_local'])
        .duplicated(subset=['building_id', 'timestamp_local'])
    )
    dup_counts = (
        gdf['is_dup'].groupby(gdf['building_id']).sum()
        .reset_index().rename(columns={'is_dup': 'duplicate_timestamps'})
    )
    dup_counts.columns = ['building_id', 'duplicate_timestamps']
    building_stats = building_stats.merge(dup_counts, on='building_id')
    building_stats['duplicate_ratio'] = (
        building_stats['duplicate_timestamps'].astype('float64')
        / building_stats['value_count'].astype('float64')
    )

    building_stats['coefficient_of_variation'] = (
        building_stats['value_std'].astype('float64')
        / building_stats['value_mean'].astype('float64')
    ).fillna(0.0)

    gdf_sorted = gdf.sort_values(['building_id', 'timestamp_local'])
    gdf_sorted['is_same'] = gdf_sorted['value'] == gdf_sorted.groupby('building_id')['value'].shift(1)
    gdf_sorted['run_group'] = (~gdf_sorted['is_same']).cumsum()
    run_lengths = gdf_sorted.groupby(['building_id', 'run_group'])['is_same'].sum().reset_index()
    run_lengths.columns = ['building_id', 'run_group', 'run_length']
    constant_threshold = int(48 / granularity_hours)
    long_runs = run_lengths[run_lengths['run_length'] >= constant_threshold]
    constant_counts = long_runs.groupby('building_id')['run_length'].count().reset_index()
    constant_counts.columns = ['building_id', 'constant_sequences']
    building_stats = building_stats.merge(constant_counts, on='building_id', how='left')
    building_stats['constant_sequences'] = building_stats['constant_sequences'].fillna(0)
    building_stats['constant_ratio'] = (
        building_stats['constant_sequences'].astype('float64')
        / building_stats['value_count'].astype('float64')
    )

    building_stats['quality_score'] = compute_quality_score(
        building_stats['completeness_ratio'],
        building_stats['coverage_ratio'],
        building_stats['continuity_ratio'],
        building_stats['zero_ratio'],
        building_stats['constant_ratio'],
        building_stats['duplicate_ratio'],
    )

    result_df = building_stats.to_pandas()
    result_df = result_df.rename(columns={
        'value_count': 'total_records',
        'value_mean': 'mean_value', 'value_std': 'std_value',
        'value_median': 'median_value', 'value_min': 'min_value', 'value_max': 'max_value',
    })

    final_cols = [
        'building_id', 'total_records', 'time_span_hours', 'time_span_days',
        'completeness_ratio', 'coverage_ratio', 'continuity_ratio',
        'gaps_2h', 'gaps_24h', 'zero_ratio', 'constant_ratio', 'duplicate_ratio',
        'mean_value', 'std_value', 'median_value', 'min_value', 'max_value',
        'coefficient_of_variation', 'quality_score',
    ]
    return result_df[final_cols].sort_values('quality_score', ascending=False)


def evaluate_all_buildings(
    df: pd.DataFrame,
    output_path: str | None = None,
    use_gpu: bool = True,
    granularity_hours: int = 1,
) -> pd.DataFrame:
    """Evaluate data quality for all buildings."""
    building_ids = df['building_id'].unique()
    logger.info("Evaluating quality for %d buildings (granularity: %dh)...", len(building_ids), granularity_hours)

    quality_df: pd.DataFrame | None = None

    if use_gpu and GPU_AVAILABLE and cudf is not None:
        try:
            df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])
            gdf = cudf.from_pandas(df)
            quality_df = evaluate_all_buildings_gpu(gdf, granularity_hours=granularity_hours)
        except (ImportError, MemoryError, RuntimeError) as e:
            logger.warning("GPU evaluation failed (%s), falling back to CPU", e)

    if quality_df is None:
        logger.info("Using CPU sequential processing...")
        quality_metrics = []
        for i, building_id in enumerate(building_ids):
            if (i + 1) % 100 == 0:
                logger.info("  Processed %d/%d buildings...", i + 1, len(building_ids))
            metrics = calculate_building_quality_metrics(df, building_id, granularity_hours=granularity_hours)
            if metrics:
                quality_metrics.append(metrics)
        quality_df = pd.DataFrame(quality_metrics).sort_values('quality_score', ascending=False)

    _log_quality_summary(quality_df)

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        quality_df.to_csv(output, index=False)
        logger.info("Quality report saved to %s", output)

    return quality_df


def _log_quality_summary(quality_df: pd.DataFrame) -> None:
    high = int((quality_df['quality_score'] >= 0.8).sum())
    medium = int(((quality_df['quality_score'] >= 0.6) & (quality_df['quality_score'] < 0.8)).sum())
    low = int((quality_df['quality_score'] < 0.6).sum())
    total = len(quality_df)

    logger.info("Quality distribution: High(>=0.8): %d (%.0f%%) | Medium: %d (%.0f%%) | Low: %d (%.0f%%)",
                high, 100 * high / total, medium, 100 * medium / total, low, 100 * low / total)


def select_high_quality_buildings(
    quality_df: pd.DataFrame,
    min_quality_score: float = 0.8,
    min_records: int = 8760,
    min_time_span_hours: int = 8760,
    max_gap_ratio: float = 0.05,
) -> list[str]:
    """Select buildings meeting quality criteria."""
    filtered = quality_df[
        (quality_df['quality_score'] >= min_quality_score)
        & (quality_df['total_records'] >= min_records)
        & (quality_df['time_span_hours'] >= min_time_span_hours)
        & (quality_df['continuity_ratio'] >= (1.0 - max_gap_ratio))
    ]

    selected = filtered['building_id'].tolist()
    logger.info("Selected %d/%d buildings (score>=%.2f, records>=%d)",
                len(selected), len(quality_df), min_quality_score, min_records)

    if len(selected) == 0:
        logger.warning("No buildings met criteria. Consider relaxing thresholds.")

    return selected


def split_buildings_for_training(
    building_ids: list[str],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_seed: int = 42,
    top_n: int | None = None,
) -> dict[str, list[str]]:
    """Split buildings into train/validation/test sets."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    if top_n is not None and top_n > 0:
        building_ids = building_ids[:top_n]

    np.random.seed(random_seed)
    ids = np.array(building_ids)
    np.random.shuffle(ids)

    n_train = int(len(ids) * train_ratio)
    n_val = int(len(ids) * val_ratio)

    split = {
        'train': ids[:n_train].tolist(),
        'validation': ids[n_train:n_train + n_val].tolist(),
        'test': ids[n_train + n_val:].tolist(),
    }

    logger.info("Split: %d train / %d val / %d test",
                len(split['train']), len(split['validation']), len(split['test']))
    return split


def save_building_split(split: dict[str, list[str]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(split, f, indent=2)
    logger.info("Building split saved to %s", path)


def load_building_split(input_path: str) -> dict[str, list[str]]:
    with open(input_path) as f:
        split = json.load(f)
    logger.info("Loaded split: %d train / %d val / %d test",
                len(split.get('train', [])), len(split.get('validation', [])), len(split.get('test', [])))
    return split


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Evaluate building data quality and create split')
    parser.add_argument('--parquet', required=True)
    parser.add_argument('--output_quality', default='data/quality_report.csv')
    parser.add_argument('--output_split', default='data/building_split.json')
    parser.add_argument('--min_quality', type=float, default=0.8)
    parser.add_argument('--min_records', type=int, default=8760)
    parser.add_argument('--min_time_span', type=int, default=8760)
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--granularity', type=int, default=1)
    parser.add_argument('--top_n', type=int, default=None)

    args = parser.parse_args()

    logger.info("Loading data from %s...", args.parquet)
    df = pd.read_parquet(args.parquet)
    logger.info("Loaded %d records, %d buildings", len(df), df['building_id'].nunique())

    quality_df = evaluate_all_buildings(
        df, args.output_quality, use_gpu=not args.no_gpu, granularity_hours=args.granularity,
    )

    selected_buildings = select_high_quality_buildings(
        quality_df, min_quality_score=args.min_quality,
        min_records=args.min_records, min_time_span_hours=args.min_time_span,
    )

    if selected_buildings:
        split = split_buildings_for_training(
            selected_buildings, train_ratio=args.train_ratio,
            val_ratio=args.val_ratio, test_ratio=args.test_ratio,
            random_seed=args.random_seed, top_n=args.top_n,
        )
        save_building_split(split, args.output_split)
    else:
        logger.warning("No buildings met criteria. Try lowering --min_quality or --min_records.")
