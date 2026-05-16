from __future__ import annotations

import argparse
import logging
import pathlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

BDG2_URL = (
    "https://media.githubusercontent.com/media/buds-lab/"
    "building-data-genome-project-2/master/data/meters/cleaned/electricity_cleaned.csv"
)


def download_raw_data(output_path: str = "data/raw_electricity.csv") -> pathlib.Path:
    """Download BDG2 raw electricity data (wide format CSV)."""
    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading BDG2 electricity data to %s...", output_file)

    try:
        response = requests.get(BDG2_URL, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_file, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc='Downloading',
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        logger.info("Downloaded %.2f MB to %s", output_file.stat().st_size / (1024 * 1024), output_file)
        return output_file

    except requests.exceptions.RequestException as e:
        logger.error("Download failed: %s", e)
        raise


def convert_to_parquet(csv_path: pathlib.Path, parquet_path: pathlib.Path) -> pd.DataFrame:
    """Convert wide format CSV to long format Parquet with explicit Arrow schema."""
    logger.info("Converting %s to long format parquet...", csv_path)

    df = pd.read_csv(csv_path)

    if "timestamp" not in df.columns:
        raise ValueError("CSV is missing 'timestamp' column")

    value_vars = [c for c in df.columns if c != "timestamp"]
    df_long = df.melt(
        id_vars=["timestamp"], value_vars=value_vars,
        var_name="building_id", value_name="value",
    )

    df_long = df_long.rename(columns={"timestamp": "timestamp_local"})
    df_long["meter"] = "electricity"

    df_long["timestamp_local"] = pd.to_datetime(df_long["timestamp_local"], errors="coerce")
    df_long = df_long.dropna(subset=["timestamp_local"])
    df_long["building_id"] = df_long["building_id"].astype(str)
    df_long["meter"] = df_long["meter"].astype(str)
    df_long["value"] = df_long["value"].astype("float32")

    df_long = df_long.sort_values(["building_id", "timestamp_local"]).reset_index(drop=True)

    schema = pa.schema([
        ('timestamp_local', pa.timestamp('ns')),
        ('building_id', pa.string()),
        ('meter', pa.string()),
        ('value', pa.float32()),
    ])

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df_long, schema=schema, preserve_index=False)
    pq.write_table(table, parquet_path, compression='snappy')

    logger.info(
        "Parquet saved: %s records, %d buildings, %s to %s",
        f"{len(df_long):,}", df_long['building_id'].nunique(),
        df_long['timestamp_local'].min(), df_long['timestamp_local'].max(),
    )

    return df_long


def main() -> None:
    parser = argparse.ArgumentParser(description='Download BDG2 electricity data and convert to parquet')
    parser.add_argument('--csv-output', type=str, default='data/raw_electricity.csv')
    parser.add_argument('--parquet-output', type=str, default='data/processed/bdg2_electricity_long.parquet')
    parser.add_argument('--skip-parquet', action='store_true', help='Skip parquet conversion')

    args = parser.parse_args()

    csv_path = download_raw_data(args.csv_output)

    if not args.skip_parquet:
        convert_to_parquet(csv_path, pathlib.Path(args.parquet_output))

    logger.info("Data ingestion complete!")


if __name__ == "__main__":
    main()
