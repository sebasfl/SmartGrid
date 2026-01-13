# src/ingest/fetch_bdg2.py
# Simple script to fetch BDG2 raw electricity data
import argparse
import pathlib
import requests
from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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

    # Convertir a tipos correctos - usar str para forzar string nativo
    df_long["building_id"] = df_long["building_id"].astype(str)
    df_long["meter"] = df_long["meter"].astype(str)
    df_long["value"] = df_long["value"].astype("float32")

    # Sort
    df_long = df_long.sort_values(["building_id", "timestamp_local"]).reset_index(drop=True)

    # Verificar tipos antes de guardar
    print(f"\n📋 Tipos antes de guardar:")
    for col in df_long.columns:
        dtype_str = str(df_long[col].dtype)
        if dtype_str == 'object':
            print(f"   {col}: {dtype_str} (será convertido a Arrow string)")
        else:
            print(f"   {col}: {dtype_str}")

    # Save parquet con schema explícito de PyArrow
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Guardando parquet con schema Arrow explícito...")

    # Definir schema explícito de PyArrow para garantizar tipos correctos
    schema = pa.schema([
        ('timestamp_local', pa.timestamp('ns')),
        ('building_id', pa.string()),  # Arrow string nativo (no large_string)
        ('meter', pa.string()),         # Arrow string nativo
        ('value', pa.float32())
    ])

    # Convertir DataFrame a Arrow Table con schema explícito
    table = pa.Table.from_pandas(df_long, schema=schema, preserve_index=False)

    # Guardar con PyArrow directamente
    pq.write_table(table, parquet_path, compression='snappy')

    # Verificar los tipos del parquet guardado
    print(f"\n📋 Verificando tipos del parquet guardado...")
    df_check = pd.read_parquet(parquet_path)
    print(f"   Tipos después de leer el parquet:")
    for col in df_check.columns:
        dtype_str = str(df_check[col].dtype)
        if dtype_str == 'string':
            print(f"     {col}: {dtype_str} ✅")
        elif 'object' in dtype_str:
            print(f"     {col}: {dtype_str} ⚠️")
        else:
            print(f"     {col}: {dtype_str}")
    
    # Verificación adicional con pyarrow
    try:
        table_verify = pq.read_table(parquet_path)
        print(f"\n📋 Schema de Arrow del archivo parquet:")
        print(f"   {table_verify.schema}")
        
        # Análisis detallado
        print(f"\n🔍 Análisis detallado de tipos Arrow:")
        for field in table_verify.schema:
            field_type = str(field.type)
            if field_type == 'string':
                print(f"   {field.name}: {field.type} ✅ (Arrow string)")
            elif 'string' in field_type.lower():
                print(f"   {field.name}: {field.type} ✅ (string)")
            else:
                print(f"   {field.name}: {field.type}")
    except ImportError:
        print("\n⚠️  PyArrow no disponible para verificación detallada del schema")
    except Exception as e:
        print(f"\n⚠️  Error al verificar schema Arrow: {e}")

    print(f"\n✅ Conversion complete!")
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
