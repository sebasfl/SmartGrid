# src/analysis/data_quality.py
# Module for evaluating data quality and selecting high-quality buildings
# Focus: Completeness, continuity, and coverage (NOT outlier removal)

import warnings
warnings.filterwarnings('ignore')

try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
    print("🚀 GPU acceleration available (CuDF + CuPy)")
    import pandas as pd
    import numpy as np
except ImportError:
    import pandas as pd
    import numpy as np
    GPU_AVAILABLE = False
    print("⚠️  GPU libraries not available, using CPU fallback")
    cudf = None
    cp = np

from typing import Dict, List
from pathlib import Path
import json


def calculate_building_quality_metrics(df: pd.DataFrame, building_id: str,
                                      granularity_hours: int = 1) -> Dict:
    """Calculate data quality metrics for a single building.

    Evaluates data completeness and consistency WITHOUT removing outliers.
    Outliers are considered legitimate anomalies for the MTL model to learn.

    Args:
        df: Full dataframe with all buildings
        building_id: Building ID to evaluate
        granularity_hours: Data granularity in hours (1 for hourly, 3 for 3-hour, etc.)

    Returns:
        Dictionary with quality metrics
    """
    building_df = df[df['building_id'] == building_id].copy()

    if len(building_df) == 0:
        return None

    # Sort by timestamp
    building_df = building_df.sort_values('timestamp_local')

    # 1. Completeness: Check for missing values
    total_rows = len(building_df)
    missing_values = building_df['value'].isna().sum()
    completeness_ratio = 1.0 - (missing_values / total_rows) if total_rows > 0 else 0.0

    # 2. Time coverage: Check if data spans expected time range
    if len(building_df) > 1:
        time_span_hours = (building_df['timestamp_local'].max() -
                          building_df['timestamp_local'].min()).total_seconds() / 3600
        expected_records = int(time_span_hours / granularity_hours) + 1
        coverage_ratio = min(1.0, total_rows / expected_records) if expected_records > 0 else 0.0
    else:
        time_span_hours = 0
        coverage_ratio = 0.0

    # 3. Temporal continuity: Check for gaps based on granularity
    if len(building_df) > 1:
        building_df['time_diff'] = building_df['timestamp_local'].diff()
        # Allow gaps up to 2x granularity (for minor clock adjustments)
        gap_threshold_hours = 2 * granularity_hours
        large_gap_threshold_hours = 24  # 24 hours is always significant
        gaps_2h = (building_df['time_diff'] > pd.Timedelta(hours=gap_threshold_hours)).sum()
        gaps_24h = (building_df['time_diff'] > pd.Timedelta(hours=large_gap_threshold_hours)).sum()
        continuity_ratio = 1.0 - (gaps_2h / (total_rows - 1)) if total_rows > 1 else 0.0
    else:
        gaps_2h = 0
        gaps_24h = 0
        continuity_ratio = 0.0

    # 4. Zero/constant values check (data quality issue, not anomalies)
    values = building_df['value'].dropna()
    if len(values) > 0:
        zero_count = (values == 0).sum()
        zero_ratio = zero_count / len(values)

        # Check for suspicious long constant sequences (>48 hours worth of intervals)
        # This indicates sensor malfunction, not real consumption patterns
        constant_sequences = 0
        current_value = None
        current_count = 0
        constant_threshold = int(48 / granularity_hours)  # 48 hours worth of intervals
        for val in values:
            if val == current_value:
                current_count += 1
                if current_count >= constant_threshold:  # Long constant period = likely broken sensor
                    constant_sequences += 1
            else:
                current_value = val
                current_count = 1
        constant_ratio = constant_sequences / len(values) if len(values) > 0 else 0.0
    else:
        zero_ratio = 1.0
        constant_ratio = 1.0

    # 5. Statistical properties (for information, not filtering)
    if len(values) > 0:
        mean_value = values.mean()
        std_value = values.std()
        median_value = values.median()
        cv = std_value / mean_value if mean_value > 0 else 0  # Coefficient of variation
        min_value = values.min()
        max_value = values.max()
    else:
        mean_value = 0
        std_value = 0
        median_value = 0
        cv = 0
        min_value = 0
        max_value = 0

    # 6. Duplicate timestamps check
    duplicate_timestamps = building_df['timestamp_local'].duplicated().sum()
    duplicate_ratio = duplicate_timestamps / total_rows if total_rows > 0 else 0.0

    # Calculate overall quality score (weighted average)
    # Focus on data integrity, NOT on outlier presence
    quality_score = (
        0.30 * completeness_ratio +       # 30% - No missing values
        0.25 * coverage_ratio +            # 25% - Good temporal coverage
        0.25 * continuity_ratio +          # 25% - No large gaps
        0.10 * (1.0 - zero_ratio) +        # 10% - Not all zeros
        0.05 * (1.0 - constant_ratio) +    # 5% - No long constant periods
        0.05 * (1.0 - duplicate_ratio)     # 5% - No duplicate timestamps
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
        'quality_score': quality_score
    }


def evaluate_all_buildings_gpu(gdf: 'cudf.DataFrame', granularity_hours: int = 1) -> pd.DataFrame:
    """GPU-accelerated vectorized quality evaluation for all buildings.

    Args:
        gdf: CuDF DataFrame with all buildings
        granularity_hours: Data granularity in hours (1 for hourly, 3 for 3-hour, etc.)

    Returns:
        Pandas DataFrame with quality metrics for each building
    """
    print(f"   Using GPU-accelerated vectorized evaluation (granularity: {granularity_hours}h)...")

    # Sort by building and timestamp
    gdf = gdf.sort_values(['building_id', 'timestamp_local'])

    # Group statistics (all computed in parallel on GPU)
    # Note: 'count' excludes NaN, 'size' includes all rows
    agg_dict = {
        'value': ['mean', 'std', 'median', 'min', 'max'],
        'timestamp_local': ['min', 'max', 'size']  # size = total records including NaN
    }

    building_stats = gdf.groupby('building_id').agg(agg_dict).reset_index()
    building_stats.columns = ['_'.join(col).strip('_') for col in building_stats.columns.values]

    # Rename size to total count (includes NaN values)
    building_stats = building_stats.rename(columns={'timestamp_local_size': 'value_count'})

    # Calculate time spans (convert to float explicitly for division)
    time_diff_seconds = (
        (building_stats['timestamp_local_max'] -
         building_stats['timestamp_local_min'])
        .astype('int64') / 1_000_000_000
    )
    building_stats['time_span_hours'] = time_diff_seconds / 3600.0
    building_stats['time_span_days'] = building_stats['time_span_hours'] / 24.0

    # Calculate completeness ratio (non-missing values)
    missing_counts = (
        gdf['value'].isna()
        .groupby(gdf['building_id'])
        .sum()
        .reset_index()
        .rename(columns={'value': 'missing_count'})
    )

    missing_counts.columns = ['building_id', 'missing_count']
    building_stats = building_stats.merge(missing_counts, on='building_id')
    building_stats['completeness_ratio'] = 1.0 - (
        building_stats['missing_count'].astype('float64') / building_stats['value_count'].astype('float64')
    )

    # Calculate coverage ratio (adjusted for granularity)
    building_stats['expected_records'] = (building_stats['time_span_hours'].astype('int64') / granularity_hours).astype('int64') + 1
    coverage_calc = building_stats['value_count'].astype('float64') / building_stats['expected_records'].astype('float64')
    building_stats['coverage_ratio'] = coverage_calc.clip(upper=1.0)

    # Calculate zero ratio
    zero_counts = (
        (gdf['value'] == 0)
        .groupby(gdf['building_id'])
        .sum()
        .reset_index()
        .rename(columns={'value': 'zero_count'})
    )

    zero_counts.columns = ['building_id', 'zero_count']
    building_stats = building_stats.merge(zero_counts, on='building_id')
    building_stats['zero_ratio'] = (
        building_stats['zero_count'].astype('float64') / building_stats['value_count'].astype('float64')
    )

    # Calculate time gaps and continuity (vectorized, adjusted for granularity)
    gdf['time_diff'] = gdf.groupby('building_id')['timestamp_local'].diff()
    # Convert to nanoseconds for comparison
    time_diff_ns = gdf['time_diff'].astype('int64')
    gap_threshold_ns = 2 * granularity_hours * 3600 * 1_000_000_000  # 2x granularity
    twentyfour_hours_ns = 24 * 3600 * 1_000_000_000

    gdf['gap_2h'] = (time_diff_ns > gap_threshold_ns).astype('int32')
    gdf['gap_24h'] = (time_diff_ns > twentyfour_hours_ns).astype('int32')

    gap_stats = gdf.groupby('building_id').agg({
        'gap_2h': 'sum',
        'gap_24h': 'sum'
    }).reset_index()
    gap_stats.columns = ['building_id', 'gaps_2h', 'gaps_24h']
    building_stats = building_stats.merge(gap_stats, on='building_id')

    building_stats['continuity_ratio'] = 1.0 - (
        building_stats['gaps_2h'].astype('float64') / (building_stats['value_count'].astype('float64') - 1.0)
    )
    building_stats['continuity_ratio'] = building_stats['continuity_ratio'].fillna(0.0)

    # cuDF-compatible duplicate detection
    gdf['is_dup'] = (
        gdf.sort_values(['building_id', 'timestamp_local'])
           .duplicated(subset=['building_id', 'timestamp_local'])
    )

    dup_counts = (
        gdf['is_dup']
        .groupby(gdf['building_id'])
        .sum()
        .reset_index()
        .rename(columns={'is_dup': 'duplicate_timestamps'})
    )

    dup_counts.columns = ['building_id', 'duplicate_timestamps']
    building_stats = building_stats.merge(dup_counts, on='building_id')
    building_stats['duplicate_ratio'] = (
        building_stats['duplicate_timestamps'].astype('float64') / building_stats['value_count'].astype('float64')
    )

    # Calculate coefficient of variation
    building_stats['coefficient_of_variation'] = (
        building_stats['value_std'].astype('float64') / building_stats['value_mean'].astype('float64')
    ).fillna(0.0)

    # Estimate constant ratio (simplified for GPU - uses zero ratio as proxy)
    # Full constant sequence detection is complex for vectorization
    building_stats['constant_ratio'] = building_stats['zero_ratio'] * 0.1  # Conservative estimate

    # Calculate overall quality score (vectorized)
    building_stats['quality_score'] = (
        0.30 * building_stats['completeness_ratio'] +
        0.25 * building_stats['coverage_ratio'] +
        0.25 * building_stats['continuity_ratio'] +
        0.10 * (1.0 - building_stats['zero_ratio']) +
        0.05 * (1.0 - building_stats['constant_ratio']) +
        0.05 * (1.0 - building_stats['duplicate_ratio'])
    )

    # Convert to pandas and rename columns
    result_df = building_stats.to_pandas()
    result_df = result_df.rename(columns={
        'value_count': 'total_records',
        'value_mean': 'mean_value',
        'value_std': 'std_value',
        'value_median': 'median_value',
        'value_min': 'min_value',
        'value_max': 'max_value'
    })

    # Select final columns
    final_cols = [
        'building_id', 'total_records', 'time_span_hours', 'time_span_days',
        'completeness_ratio', 'coverage_ratio', 'continuity_ratio',
        'gaps_2h', 'gaps_24h', 'zero_ratio', 'constant_ratio', 'duplicate_ratio',
        'mean_value', 'std_value', 'median_value', 'min_value', 'max_value',
        'coefficient_of_variation', 'quality_score'
    ]

    result_df = result_df[final_cols]
    result_df = result_df.sort_values('quality_score', ascending=False)

    return result_df


def evaluate_all_buildings(df: pd.DataFrame, output_path: str = None,
                           use_gpu: bool = True, granularity_hours: int = 1) -> pd.DataFrame:
    """Evaluate data quality for all buildings in the dataset.

    Args:
        df: Full dataframe with all buildings
        output_path: Optional path to save quality report as CSV
        use_gpu: Use GPU acceleration if available (default: True)
        granularity_hours: Data granularity in hours (1 for hourly, 3 for 3-hour, etc.)

    Returns:
        DataFrame with quality metrics for each building
    """
    print(f"\n📊 Evaluating data quality for all buildings (granularity: {granularity_hours}h)...")

    building_ids = df['building_id'].unique()
    n_buildings = len(building_ids)
    print(f"   Total buildings: {n_buildings}")

    # Try GPU-accelerated evaluation first
    if use_gpu and GPU_AVAILABLE and cudf is not None:
        try:
            # Convert to CuDF for GPU processing
            print("   Converting to GPU dataframe...")
            df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])
            gdf = cudf.from_pandas(df)

            # GPU-accelerated evaluation
            quality_df = evaluate_all_buildings_gpu(gdf, granularity_hours=granularity_hours)

            print(f"   ✅ GPU evaluation complete!")

        except Exception as e:
            print(f"   ⚠️  GPU evaluation failed ({e}), falling back to CPU...")
            use_gpu = False

    # CPU fallback (original sequential processing)
    if not use_gpu or not GPU_AVAILABLE:
        print("   Using CPU sequential processing...")
        quality_metrics = []

        for i, building_id in enumerate(building_ids):
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{n_buildings} buildings...")

            metrics = calculate_building_quality_metrics(df, building_id, granularity_hours=granularity_hours)
            if metrics:
                quality_metrics.append(metrics)

        quality_df = pd.DataFrame(quality_metrics)
        quality_df = quality_df.sort_values('quality_score', ascending=False)

    print(f"\n✅ Quality evaluation complete for {len(quality_df)} buildings")
    print(f"\n📈 Quality Score Statistics:")
    print(f"   Mean: {quality_df['quality_score'].mean():.3f}")
    print(f"   Median: {quality_df['quality_score'].median():.3f}")
    print(f"   Std: {quality_df['quality_score'].std():.3f}")
    print(f"   Min: {quality_df['quality_score'].min():.3f}")
    print(f"   Max: {quality_df['quality_score'].max():.3f}")

    # Show distribution by quality tiers
    high_quality = (quality_df['quality_score'] >= 0.8).sum()
    medium_quality = ((quality_df['quality_score'] >= 0.6) &
                     (quality_df['quality_score'] < 0.8)).sum()
    low_quality = (quality_df['quality_score'] < 0.6).sum()

    print(f"\n📊 Quality Distribution:")
    print(f"   High (≥0.8):    {high_quality} buildings ({100*high_quality/len(quality_df):.1f}%)")
    print(f"   Medium (0.6-0.8): {medium_quality} buildings ({100*medium_quality/len(quality_df):.1f}%)")
    print(f"   Low (<0.6):     {low_quality} buildings ({100*low_quality/len(quality_df):.1f}%)")

    # Show top 10 highest quality buildings
    print(f"\n🏆 Top 10 Highest Quality Buildings:")
    display_cols = ['building_id', 'quality_score', 'completeness_ratio',
                   'continuity_ratio', 'time_span_days', 'total_records']
    print(quality_df[display_cols].head(10).to_string(index=False))

    # Save report if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        quality_df.to_csv(output_path, index=False)
        print(f"\n💾 Quality report saved to {output_path}")

    return quality_df


def select_high_quality_buildings(quality_df: pd.DataFrame,
                                  min_quality_score: float = 0.8,
                                  min_records: int = 8760,  # 1 year of hourly data
                                  min_time_span_hours: int = 8760,
                                  max_gap_ratio: float = 0.05) -> List[str]:
    """Select buildings that meet quality criteria.

    Args:
        quality_df: DataFrame with quality metrics (from evaluate_all_buildings)
        min_quality_score: Minimum quality score threshold (0-1)
        min_records: Minimum number of data records
        min_time_span_hours: Minimum time span in hours
        max_gap_ratio: Maximum allowed ratio of gaps (1-continuity_ratio)

    Returns:
        List of building IDs that meet quality criteria
    """
    filtered = quality_df[
        (quality_df['quality_score'] >= min_quality_score) &
        (quality_df['total_records'] >= min_records) &
        (quality_df['time_span_hours'] >= min_time_span_hours) &
        (quality_df['continuity_ratio'] >= (1.0 - max_gap_ratio))
    ]

    selected_buildings = filtered['building_id'].tolist()

    print(f"\n🔍 Building Selection Criteria:")
    print(f"   Min Quality Score: {min_quality_score}")
    print(f"   Min Records: {min_records}")
    print(f"   Min Time Span: {min_time_span_hours} hours ({min_time_span_hours/24:.0f} days)")
    print(f"   Max Gap Ratio: {max_gap_ratio*100:.1f}%")
    print(f"\n✅ Selected {len(selected_buildings)}/{len(quality_df)} buildings")
    print(f"   Selection rate: {100*len(selected_buildings)/len(quality_df):.1f}%")

    if len(selected_buildings) > 0:
        selected_quality = filtered['quality_score']
        print(f"\n📊 Selected Buildings Quality Stats:")
        print(f"   Mean Quality: {selected_quality.mean():.3f}")
        print(f"   Median Quality: {selected_quality.median():.3f}")
        print(f"   Min Quality: {selected_quality.min():.3f}")
        print(f"   Mean Completeness: {filtered['completeness_ratio'].mean():.3f}")
        print(f"   Mean Continuity: {filtered['continuity_ratio'].mean():.3f}")
    else:
        print("\n⚠️  No buildings met all criteria. Consider relaxing thresholds.")

    return selected_buildings


def split_buildings_for_training(building_ids: List[str],
                                 train_ratio: float = 0.6,
                                 val_ratio: float = 0.2,
                                 test_ratio: float = 0.2,
                                 random_seed: int = 42,
                                 top_n: int = None) -> Dict[str, List[str]]:
    """Split high-quality buildings into train/validation/test sets.

    Args:
        building_ids: List of building IDs to split (should be pre-sorted by quality)
        train_ratio: Ratio of buildings for training (default: 0.6)
        val_ratio: Ratio of buildings for validation (default: 0.2)
        test_ratio: Ratio of buildings for test (default: 0.2)
        random_seed: Random seed for reproducibility
        top_n: If specified, only use top N buildings (e.g., 10 for top 10)

    Returns:
        Dictionary with 'train', 'validation', and 'test' building lists
    """
    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    # Select top N buildings if specified
    if top_n is not None and top_n > 0:
        building_ids = building_ids[:top_n]
        print(f"\n🏆 Using top {top_n} buildings only")

    np.random.seed(random_seed)
    building_ids_array = np.array(building_ids)
    np.random.shuffle(building_ids_array)

    n_total = len(building_ids_array)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    split = {
        'train': building_ids_array[:n_train].tolist(),
        'validation': building_ids_array[n_train:n_train + n_val].tolist(),
        'test': building_ids_array[n_train + n_val:].tolist()
    }

    print(f"\n📂 Building Split:")
    print(f"   Training buildings:   {len(split['train'])} ({100*len(split['train'])/n_total:.1f}%)")
    print(f"   Validation buildings: {len(split['validation'])} ({100*len(split['validation'])/n_total:.1f}%)")
    print(f"   Test buildings:       {len(split['test'])} ({100*len(split['test'])/n_total:.1f}%)")
    print(f"   Random seed: {random_seed}")

    return split


def save_building_split(split: Dict[str, List[str]], output_path: str):
    """Save building split to JSON file.

    Args:
        split: Dictionary with building splits
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(split, f, indent=2)

    print(f"\n💾 Building split saved to {output_path}")


def load_building_split(input_path: str) -> Dict[str, List[str]]:
    """Load building split from JSON file.

    Args:
        input_path: Path to JSON file

    Returns:
        Dictionary with building splits
    """
    with open(input_path, 'r') as f:
        split = json.load(f)

    print(f"\n📂 Loaded building split from {input_path}")
    print(f"   Training buildings:   {len(split.get('train', []))}")
    print(f"   Validation buildings: {len(split.get('validation', []))}")
    print(f"   Test buildings:       {len(split.get('test', []))}")

    return split


if __name__ == "__main__":
    """Example usage of data quality evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate building data quality and create train/holdout split')
    parser.add_argument('--parquet', required=True, help='Path to preprocessed parquet file')
    parser.add_argument('--output_quality', default='data/quality_report.csv',
                       help='Path to save quality report')
    parser.add_argument('--output_split', default='data/building_split.json',
                       help='Path to save building split')
    parser.add_argument('--min_quality', type=float, default=0.8,
                       help='Minimum quality score (0-1)')
    parser.add_argument('--min_records', type=int, default=8760,
                       help='Minimum number of records (default: 1 year = 8760 hours)')
    parser.add_argument('--min_time_span', type=int, default=8760,
                       help='Minimum time span in hours (default: 1 year)')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                       help='Ratio of buildings for training (default: 0.6)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Ratio of buildings for validation (default: 0.2)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                       help='Ratio of buildings for test (default: 0.2)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--no_gpu', action='store_true',
                       help='Disable GPU acceleration (use CPU only)')
    parser.add_argument('--granularity', type=int, default=1,
                       help='Data granularity in hours (1 for hourly, 3 for 3-hour resampled)')
    parser.add_argument('--top_n', type=int, default=None,
                       help='Use only top N buildings with highest quality (e.g., 10 for top 10)')

    args = parser.parse_args()

    # Load data
    print(f"\n📥 Loading data from {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    print(f"   Loaded {len(df)} records, {df['building_id'].nunique()} buildings")

    # Evaluate quality (GPU by default unless --no_gpu flag is set)
    quality_df = evaluate_all_buildings(df, args.output_quality, use_gpu=not args.no_gpu,
                                       granularity_hours=args.granularity)

    # Select high-quality buildings
    selected_buildings = select_high_quality_buildings(
        quality_df,
        min_quality_score=args.min_quality,
        min_records=args.min_records,
        min_time_span_hours=args.min_time_span
    )

    # Split buildings
    if len(selected_buildings) > 0:
        split = split_buildings_for_training(
            selected_buildings,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.random_seed,
            top_n=args.top_n
        )
        save_building_split(split, args.output_split)

        print(f"\n✅ Process complete!")
        print(f"   Quality report: {args.output_quality}")
        print(f"   Building split: {args.output_split}")
    else:
        print("\n⚠️  No buildings met the quality criteria!")
        print("   Try lowering --min_quality or --min_records thresholds")
