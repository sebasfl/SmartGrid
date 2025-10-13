# src/ingest/fetch_bdg2.py
# Simple script to fetch BDG2 raw electricity data
import argparse
import pathlib
import requests
from tqdm import tqdm
import pandas as pd


def download_raw_data(output_path: str = "data/raw_electricity.csv"):
    """
    Download BDG2 raw electricity data (wide format CSV).

    Args:
        output_path: Path where to save the CSV file (default: data/raw_electricity.csv)
    """
    # URL to BDG2 cleaned electricity data (wide format)
    url = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/meters/cleaned/electricity_cleaned.csv"

    output_file = pathlib.Path(output_path)

    # Create data directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading BDG2 electricity data...")
    print(f"   URL: {url}")
    print(f"   Output: {output_file}")

    try:
        # Download with progress bar
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_file, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc='Downloading'
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"\n✅ Download complete!")
        print(f"   File saved to: {output_file.absolute()}")
        print(f"   Size: {output_file.stat().st_size / (1024*1024):.2f} MB")

        return output_file

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Download failed: {e}")
        raise


def convert_to_parquet(csv_path: pathlib.Path, parquet_path: pathlib.Path):
    """
    Convert wide format CSV to long format Parquet.

    Wide format: columns = ['timestamp', 'building_1', 'building_2', ...]
    Long format: columns = ['timestamp_local', 'building_id', 'meter', 'value']

    Args:
        csv_path: Path to wide format CSV
        parquet_path: Path to save long format parquet
    """
    print(f"\n📊 Converting to long format parquet...")
    print(f"   Reading: {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)

    if "timestamp" not in df.columns:
        raise ValueError("CSV is missing 'timestamp' column")

    # Melt wide -> long
    value_vars = [c for c in df.columns if c != "timestamp"]
    df_long = df.melt(
        id_vars=["timestamp"],
        value_vars=value_vars,
        var_name="building_id",
        value_name="value"
    )

    # Rename and add columns
    df_long = df_long.rename(columns={"timestamp": "timestamp_local"})
    df_long["meter"] = "electricity"

    # Clean types
    df_long["timestamp_local"] = pd.to_datetime(df_long["timestamp_local"], errors="coerce")
    df_long = df_long.dropna(subset=["timestamp_local"])
    df_long["building_id"] = df_long["building_id"].astype(str)

    # Sort
    df_long = df_long.sort_values(["building_id", "timestamp_local"]).reset_index(drop=True)

    # Save parquet
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_parquet(parquet_path, index=False)

    print(f"✅ Conversion complete!")
    print(f"   Parquet saved to: {parquet_path.absolute()}")
    print(f"   Records: {len(df_long):,}")
    print(f"   Buildings: {df_long['building_id'].nunique()}")
    print(f"   Date range: {df_long['timestamp_local'].min()} to {df_long['timestamp_local'].max()}")

    return df_long


def main():
    parser = argparse.ArgumentParser(
        description='Download BDG2 electricity data and convert to parquet'
    )
    parser.add_argument(
        '--csv-output',
        type=str,
        default='data/raw_electricity.csv',
        help='Output path for CSV file (default: data/raw_electricity.csv)'
    )
    parser.add_argument(
        '--parquet-output',
        type=str,
        default='data/processed/bdg2_electricity_long.parquet',
        help='Output path for parquet file (default: data/processed/bdg2_electricity_long.parquet)'
    )
    parser.add_argument(
        '--skip-parquet',
        action='store_true',
        help='Skip parquet conversion (only download CSV)'
    )

    args = parser.parse_args()

    # Download CSV
    csv_path = download_raw_data(args.csv_output)

    # Convert to parquet (unless skipped)
    if not args.skip_parquet:
        parquet_path = pathlib.Path(args.parquet_output)
        convert_to_parquet(csv_path, parquet_path)
    else:
        print("\n⏭️  Skipping parquet conversion (--skip-parquet flag)")

    print(f"\n🎉 Data ingestion complete!")


if __name__ == "__main__":
    main()
