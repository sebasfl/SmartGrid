# src/analysis/clean_building_data_parallel.py
# Simplified preprocessing: Time features + Deduplication only
import argparse
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
    cudf = pd
    cp = np

from pathlib import Path


class BasicPreprocessor:
    """Basic preprocessing: time features and deduplication."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.gpu_available = GPU_AVAILABLE

    def load_data(self, parquet_path: Path):
        """Load parquet data with GPU acceleration if available."""
        if self.verbose:
            print(f"Loading data {'on GPU' if self.gpu_available else 'on CPU'}...")

        # Load with pandas first
        df = pd.read_parquet(parquet_path)

        # Ensure we have the right columns
        ts_col = "timestamp_local" if "timestamp_local" in df.columns else "timestamp_utc"
        if "meter" in df.columns:
            df = df[df["meter"] == "electricity"]

        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df[ts_col] = pd.to_datetime(df[ts_col])

        # Rename timestamp column for consistency
        if ts_col != "timestamp_local":
            df = df.rename(columns={ts_col: "timestamp_local"})

        df = df.sort_values(["building_id", "timestamp_local"]).reset_index(drop=True)

        # Convert to GPU if available
        if self.gpu_available:
            try:
                if self.verbose:
                    print(f"   Transferring to GPU...")
                df = cudf.from_pandas(df)
                if self.verbose:
                    print(f"   ✅ Data loaded on GPU")
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  GPU transfer failed: {e}")
                    print(f"   Using CPU mode")
                self.gpu_available = False

        return df

    def add_time_features(self, df):
        """Add vectorized time features."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"⏰ STEP 1/2: Adding time features (vectorized)...")
            print(f"{'='*60}")

        if self.gpu_available:
            # CuDF timestamp handling
            if not hasattr(df['timestamp_local'].dtype, 'tz'):
                df['timestamp_local'] = cudf.to_datetime(df['timestamp_local'])

            # Extract time features using CuDF
            df['hour'] = df['timestamp_local'].dt.hour.astype('int8')
            df['day_of_week'] = df['timestamp_local'].dt.dayofweek.astype('int8')
            df['month'] = df['timestamp_local'].dt.month.astype('int8')
            df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
            df['is_working_hours'] = (
                (df['hour'] >= 8) & (df['hour'] <= 18) & (df['is_weekend'] == 0)
            ).astype('int8')

            # Seasonal features
            df['quarter'] = df['timestamp_local'].dt.quarter.astype('int8')
            df['day_of_year'] = df['timestamp_local'].dt.dayofyear.astype('int16')
        else:
            # Pandas path
            df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])

            # Extract all time features at once (vectorized)
            df['hour'] = df['timestamp_local'].dt.hour.astype('int8')
            df['day_of_week'] = df['timestamp_local'].dt.dayofweek.astype('int8')
            df['month'] = df['timestamp_local'].dt.month.astype('int8')
            df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
            df['is_working_hours'] = (
                (df['hour'] >= 8) & (df['hour'] <= 18) & (df['is_weekend'] == 0)
            ).astype('int8')

            # Seasonal features
            df['quarter'] = df['timestamp_local'].dt.quarter.astype('int8')
            df['day_of_year'] = df['timestamp_local'].dt.dayofyear.astype('int16')

        if self.verbose:
            print(f"\n✅ Time features added:")
            print(f"   • 7 features: hour, day_of_week, month, is_weekend, is_working_hours, quarter, day_of_year")

        return df

    def remove_duplicates(self, df):
        """Remove duplicate timestamps per building."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🔍 STEP 2/2: Removing duplicate timestamps...")
            print(f"{'='*60}")

        before_dedup = len(df)
        df = df.drop_duplicates(subset=['building_id', 'timestamp_local'], keep='first')
        after_dedup = len(df)

        if self.verbose:
            duplicates_removed = before_dedup - after_dedup
            print(f"\n✅ Deduplication completed:")
            print(f"   • Removed {duplicates_removed:,} duplicate records")

        return df

    def process(self, df):
        """Run complete preprocessing pipeline."""
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🚀 STARTING BASIC PREPROCESSING")
            print(f"{'='*80}")
            print(f"📦 Input: {len(df):,} records, {df['building_id'].nunique():,} buildings")
            print(f"⚡ Acceleration: {'GPU (CuDF)' if self.gpu_available else 'CPU (Pandas)'}")
            print(f"{'='*80}\n")

        original_size = len(df)

        # Step 1: Add time features
        df = self.add_time_features(df)

        # Step 2: Remove duplicates
        df = self.remove_duplicates(df)

        final_size = len(df)

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"✅ PREPROCESSING COMPLETED!")
            print(f"{'='*80}")
            print(f"📊 Retention rate: {(final_size/original_size)*100:.2f}% ({original_size:,} → {final_size:,} records)")
            print(f"🏢 Final buildings: {df['building_id'].nunique():,}")
            print(f"{'='*80}\n")

        return df


def main():
    parser = argparse.ArgumentParser(description='Basic data preprocessing: time features + deduplication')
    parser.add_argument('--parquet', type=Path, required=True, help='Input parquet file')
    parser.add_argument('--output', type=Path, required=True, help='Output cleaned parquet file')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"🚀 INITIALIZING BASIC PREPROCESSING PIPELINE")
    print(f"{'='*80}\n")

    processor = BasicPreprocessor(verbose=True)

    # Load data
    df = processor.load_data(args.parquet)
    print(f"✅ Loaded {len(df):,} records for {df['building_id'].nunique():,} buildings\n")

    # Process
    df_clean = processor.process(df)

    # Save cleaned dataset
    print("💾 Saving cleaned dataset...")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if processor.gpu_available:
        # Convert to pandas for saving if using CuDF
        df_clean.to_pandas().to_parquet(args.output, index=False)
    else:
        df_clean.to_parquet(args.output, index=False)

    print(f"\n🎉 Success! Preprocessing completed!")
    print(f"📈 Final dataset: {len(df_clean):,} records, {df_clean['building_id'].nunique():,} buildings")
    print(f"💾 Clean data saved to: {args.output}")


if __name__ == "__main__":
    main()
