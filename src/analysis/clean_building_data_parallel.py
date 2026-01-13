# src/analysis/clean_building_data_parallel.py
# Simplified preprocessing: Time features + Deduplication only
import argparse
import warnings
import os
warnings.filterwarnings('ignore')

# Set Numba environment variable before importing CuDF
os.environ['NUMBA_CUDA_ENABLE_PYNVJITLINK'] = '1'

try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
    print("🚀 GPU acceleration: cuDF + CuPy available")
except Exception as e:
    print(f"⚠️ GPU libraries unavailable: {e}")
    GPU_AVAILABLE = False

import pandas as pd
import numpy as np


from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


class BasicPreprocessor:
    """Basic preprocessing: time features and deduplication."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.gpu_available = GPU_AVAILABLE

    def load_data(self, parquet_path: Path):
        """Load parquet data with GPU acceleration if available."""
        if self.verbose:
            print(f"Loading data {'on GPU' if self.gpu_available else 'on CPU'}...")
    
        # Load data
        if self.gpu_available:
            df = cudf.read_parquet(parquet_path)
        else:
            df = pd.read_parquet(parquet_path)
    
        # Ensure we have the right columns
        ts_col = "timestamp_local" if "timestamp_local" in df.columns else "timestamp_utc"
    
        # Filter only electricity meter
        if "meter" in df.columns:
            df = df[df["meter"] == "electricity"]
    
        if self.gpu_available:
            df["building_id"] = df["building_id"].astype("string")
            if "meter" in df.columns:
                df["meter"] = df["meter"].astype("string")
        else:
            df["building_id"] = df["building_id"].astype(str)
            if "meter" in df.columns:
                df["meter"] = df["meter"].astype(str)

        
        # Numeric + timestamp casting
        df["value"] = df["value"].astype("float32")
        if self.gpu_available:
            df["timestamp_local"] = cudf.to_datetime(df["timestamp_local"])
        else:
            df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])

    
        # Rename timestamp column for consistency
        if ts_col != "timestamp_local":
            df = df.rename(columns={ts_col: "timestamp_local"})
    
        # Sort
        df = df.sort_values(["building_id", "timestamp_local"]).reset_index(drop=True)
    
        # Verificación final
        if self.verbose and self.gpu_available:
            print("\n📋 Verificación final de tipos en load_data:")
            for col in df.columns:
                dtype_str = str(df[col].dtype)
                if dtype_str == 'object':
                    print(f"   {col}: {dtype_str} ⚠️ (NO optimizado para GPU)")
                elif 'string' in dtype_str:
                    print(f"   {col}: {dtype_str} ✅ (Optimizado para GPU)")
                else:
                    print(f"   {col}: {dtype_str}")
    
        return df

    def add_time_features(self, df):
        """Add vectorized time features."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"⏰ STEP 1/2: Adding time features (vectorized)...")
            print(f"{'='*60}")

        if self.gpu_available:
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

    def resample_to_3h(self, df):
        """Resample data from 1-hour to 3-hour intervals using GPU-accelerated operations."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📊 STEP 2/3: Resampling from 1h to 3h intervals...")
            print(f"{'='*60}")

        before_resample = len(df)

        if self.gpu_available:
            # GPU-accelerated approach using CuDF
            print(f"   🚀 Using GPU-accelerated resampling (CuDF)")

            # Create 3-hour time bins using floor division
            # Convert timestamp to hours since epoch, then divide by 3 and multiply back
            df['time_bin'] = (df['timestamp_local'].astype('int64') // (3 * 3600 * 1_000_000_000)) * (3 * 3600 * 1_000_000_000)
            df['time_bin'] = df['time_bin'].astype('datetime64[ns]')

            # Group by building and time bin, aggregate with mean
            agg_dict = {'value': 'mean'}

            # Keep first occurrence of categorical columns
            if 'meter' in df.columns:
                agg_dict['meter'] = 'first'

            df_resampled = df.groupby(['building_id', 'time_bin'], as_index=False).agg(agg_dict)

            # Rename time_bin back to timestamp_local
            df_resampled = df_resampled.rename(columns={'time_bin': 'timestamp_local'})

            # Drop NaN values
            df_resampled = df_resampled.dropna(subset=['value'])

            df = df_resampled

        else:
            # CPU fallback using pandas
            print(f"   💻 Using CPU resampling (Pandas)")

            # Create 3-hour time bins
            df['time_bin'] = df['timestamp_local'].dt.floor('3h')

            # Group by building and time bin
            agg_dict = {'value': 'mean'}
            if 'meter' in df.columns:
                agg_dict['meter'] = 'first'

            df_resampled = df.groupby(['building_id', 'time_bin'], as_index=False).agg(agg_dict)

            # Rename time_bin back to timestamp_local
            df_resampled = df_resampled.rename(columns={'time_bin': 'timestamp_local'})

            # Drop NaN values
            df_resampled = df_resampled.dropna(subset=['value'])

            df = df_resampled

        after_resample = len(df)

        if self.verbose:
            print(f"\n✅ Resampling completed:")
            print(f"   • Original records: {before_resample:,}")
            print(f"   • After 3h resampling: {after_resample:,}")
            print(f"   • Reduction: {(1 - after_resample/before_resample)*100:.1f}%")
            print(f"   • Speed: {'GPU-accelerated ⚡' if self.gpu_available else 'CPU'}")

        return df

    def remove_duplicates(self, df):
        """Remove duplicate timestamps per building."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🔍 STEP 3/3: Removing duplicate timestamps...")
            print(f"{'='*60}")

        before_dedup = len(df)
        df = df.drop_duplicates(subset=['building_id', 'timestamp_local'], keep='first')
        after_dedup = len(df)

        if self.verbose:
            duplicates_removed = before_dedup - after_dedup
            print(f"\n✅ Deduplication completed:")
            print(f"   • Removed {duplicates_removed:,} duplicate records")

        return df

    def process(self, df, skip_deduplication=False, resample_3h=False):
        """Run complete preprocessing pipeline.

        Args:
            df: Input dataframe
            skip_deduplication: If True, skip duplicate removal (useful for data quality analysis)
            resample_3h: If True, resample data from 1h to 3h intervals
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🚀 STARTING BASIC PREPROCESSING")
            print(f"{'='*80}")
            print(f"📦 Input: {len(df):,} records, {df['building_id'].nunique():,} buildings")
            print(f"⚡ Acceleration: {'GPU (CuDF)' if self.gpu_available else 'CPU (Pandas)'}")
            if skip_deduplication:
                print(f"⚠️  Deduplication DISABLED (for data quality analysis)")
            if resample_3h:
                print(f"📊 3-hour resampling ENABLED (memory optimization)")
            print(f"{'='*80}\n")

        original_size = len(df)

        # Step 1: Add time features (before resampling to preserve hourly features)
        df = self.add_time_features(df)

        # Step 2: Resample to 3h intervals (optional)
        if resample_3h:
            df = self.resample_to_3h(df)
            # Re-add time features after resampling
            df = self.add_time_features(df)

        # Step 3: Remove duplicates (unless skipped)
        if not skip_deduplication:
            df = self.remove_duplicates(df)
        else:
            if self.verbose:
                print(f"\n⚠️  SKIPPING deduplication step")
                print(f"   (Duplicates will be counted in data quality evaluation)")

        final_size = len(df)

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"✅ PREPROCESSING COMPLETED!")
            print(f"{'='*80}")
            print(f"📊 Retention rate: {(final_size/original_size)*100:.2f}% ({original_size:,} → {final_size:,} records)")
            print(f"🏢 Final buildings: {df['building_id'].nunique():,}")
            print(f"{'='*80}\n")
            
        if self.verbose and self.gpu_available:
            print("\n📋 Verificación final de tipos:")
            for col in df.columns:
                print(f"   {col}: {df[col].dtype}")

        return df


def main():
    parser = argparse.ArgumentParser(description='Basic data preprocessing: time features + deduplication')
    parser.add_argument('--parquet', type=Path, required=True, help='Input parquet file')
    parser.add_argument('--output', type=Path, required=True, help='Output cleaned parquet file')
    parser.add_argument('--no-deduplicate', action='store_true',
                       help='Skip deduplication step (useful for data quality analysis)')
    parser.add_argument('--resample-3h', action='store_true',
                       help='Resample data from 1h to 3h intervals (reduces memory usage by ~66%%)')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"🚀 INITIALIZING BASIC PREPROCESSING PIPELINE")
    print(f"{'='*80}\n")

    processor = BasicPreprocessor(verbose=True)

    # Load data
    df = processor.load_data(args.parquet)
    print(f"✅ Loaded {len(df):,} records for {df['building_id'].nunique():,} buildings\n")

    # Process
    df_clean = processor.process(df, skip_deduplication=args.no_deduplicate, resample_3h=args.resample_3h)

    # Save cleaned dataset
    print("💾 Saving cleaned dataset...")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if processor.gpu_available:
        # Para GPU: Convertir a pandas con tipos correctos y luego a Arrow
        print("🔄 Convirtiendo de CuDF a Arrow con schema explícito...")

        # Convertir a pandas primero
        df_pandas = df_clean.to_pandas()

        # Determinar columnas presentes
        cols = df_pandas.columns.tolist()
        schema_fields = [
            ('timestamp_local', pa.timestamp('ns')),
            ('building_id', pa.string()),
            ('value', pa.float32())
        ]

        # Agregar 'meter' si existe
        if 'meter' in cols:
            schema_fields.insert(2, ('meter', pa.string()))

        # Agregar columnas de tiempo (si existen)
        time_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_working_hours', 'quarter', 'day_of_year']
        for tc in time_cols:
            if tc in cols:
                if tc.startswith('is_'):
                    schema_fields.append((tc, pa.bool_()))
                else:
                    schema_fields.append((tc, pa.int32()))

        schema = pa.schema(schema_fields)

        # Convertir a Arrow Table con schema explícito
        table = pa.Table.from_pandas(df_pandas, schema=schema, preserve_index=False)

        # Guardar con PyArrow
        pq.write_table(table, args.output, compression='snappy')
        print(f"✅ Guardado con schema Arrow: {schema}")
    else:
        # Para CPU: Usar pandas con Arrow schema
        print("🔄 Convirtiendo a Arrow con schema explícito...")

        # Determinar columnas presentes
        cols = df_clean.columns.tolist()
        schema_fields = [
            ('timestamp_local', pa.timestamp('ns')),
            ('building_id', pa.string()),
            ('value', pa.float32())
        ]

        # Agregar 'meter' si existe
        if 'meter' in cols:
            schema_fields.insert(2, ('meter', pa.string()))

        # Agregar columnas de tiempo (si existen)
        time_cols = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_working_hours', 'quarter', 'day_of_year']
        for tc in time_cols:
            if tc in cols:
                if tc.startswith('is_'):
                    schema_fields.append((tc, pa.bool_()))
                else:
                    schema_fields.append((tc, pa.int32()))

        schema = pa.schema(schema_fields)

        # Convertir a Arrow Table con schema explícito
        table = pa.Table.from_pandas(df_clean, schema=schema, preserve_index=False)

        # Guardar con PyArrow
        pq.write_table(table, args.output, compression='snappy')
        print(f"✅ Guardado con schema Arrow: {schema}")


    print(f"\n🎉 Success! Preprocessing completed!")
    print(f"📈 Final dataset: {len(df_clean):,} records, {df_clean['building_id'].nunique():,} buildings")
    print(f"💾 Clean data saved to: {args.output}")


if __name__ == "__main__":
    main()
