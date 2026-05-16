# SmartGrid - Hybrid CNN-LSTM for Energy Consumption Forecasting

A complete **Hybrid CNN-LSTM** pipeline for energy consumption forecasting using the Building Data Genome Project 2 (BDG2) dataset. Features GPU-accelerated preprocessing, mixed-precision training, and batch inference.

## Project Overview

SmartGrid implements an end-to-end deep learning pipeline that:
- **Forecasts** energy consumption 30 days ahead (multi-step regression)
- Uses a **Hybrid CNN-LSTM** architecture (CNN for local features + LSTM for temporal patterns)
- Supports **3-hour resampled data** for memory-efficient training on GPUs with <12GB VRAM
- Processes **27.7M records** from **1,578 buildings** (or ~9.2M with 3h resampling)

## Architecture

### Hybrid CNN-LSTM Model

```
Input Sequence → CNN (local features) → LSTM (temporal patterns) → Dense Head → Forecast
```

- **CNN Feature Extractor**: 3 convolutional layers [64, 128, 128] with BatchNorm, MaxPooling, Dropout
- **LSTM Temporal Encoder**: 2 bidirectional LSTM layers [128, 64] (generic GPU kernel for cuDNN 9.0+ compatibility)
- **Forecasting Head**: 2 dense layers [128, 64] → Dense(horizon)
- **Parameters**: ~500K-1M trainable (depending on configuration)

### Training Configuration
- **Loss**: MSE (Mean Squared Error) for forecasting
- **Optimizer**: Adam (LR=1e-4, gradient clip=1.0)
- **Mixed Precision**: FP16 for 2x speedup
- **LSTM Implementation**: Generic GPU kernel (cuDNN 9.0+ compatibility)
- **Callbacks**: Early stopping (patience=10), model checkpointing, LR plateau reduction
- **Data Split**: 60% train, 20% val, 20% test (by buildings, via data quality analysis)
- **Sequences**: Lookback=480 intervals (60 days), Horizon=240 intervals (30 days) with 3h data

## Quick Start

### Prerequisites
- Docker with NVIDIA GPU support
- NVIDIA Container Toolkit installed
- GPU with 4-6GB+ VRAM (GTX 1650 Ti or better with 3h resampled data)

### Complete Pipeline (Step-by-Step)

#### 1. Build GPU Container
```bash
docker compose build --no-cache trainer-gpu
```

#### 2. Data Ingestion
```bash
# Fetch BDG2 data and convert to parquet
docker compose run --rm trainer-gpu python -m src.ingest.fetch_bdg2
```

#### 3. Preprocessing (Time Features + Deduplication + Optional 3h Resampling)
```bash
# WITH 3-hour resampling (RECOMMENDED - reduces memory by 66%)
docker compose run --rm trainer-gpu python -m src.analysis.clean_building_data_parallel \
  --parquet data/processed/bdg2_electricity_long.parquet \
  --output data/processed/bdg2_cleaned_3h.parquet \
  --resample-3h

# WITHOUT resampling (original 1-hour granularity - requires 12GB+ VRAM)
docker compose run --rm trainer-gpu python -m src.analysis.clean_building_data_parallel \
  --parquet data/processed/bdg2_electricity_long.parquet \
  --output data/processed/bdg2_cleaned.parquet
```

#### 4. Data Quality Analysis & Building Selection (Recommended)
```bash
# Select top 50 high-quality buildings with train/val/test split (60/20/20)
docker compose run --rm trainer-gpu python -m src.analysis.data_quality \
  --parquet data/processed/bdg2_cleaned_3h.parquet \
  --output_quality data/quality_report_3h.csv \
  --output_split data/building_split_top_n.json \
  --min_quality 0.99 --min_records 2920 --granularity 3 --top_n 50
```

#### 5. Train CNN-LSTM Model
```bash
# With 3h data + high-quality buildings (RECOMMENDED)
docker compose run --rm trainer-gpu python -m src.main \
  --parquet data/processed/bdg2_cleaned_3h.parquet \
  --building_split data/building_split_top_n.json \
  --model_dir models/cnn_lstm_3h_quality \
  --epochs 50 --use_full_dataset

# Quick test (2 epochs)
docker compose run --rm trainer-gpu python -m src.main \
  --parquet data/processed/bdg2_cleaned_3h.parquet \
  --building_split data/building_split_top_n.json \
  --model_dir models/cnn_lstm_3h_quality \
  --epochs 2 --use_full_dataset
```

#### 6. Batch Inference (Prediction)
```bash
docker compose run --rm trainer-gpu python -m src.predict \
  --model_dir models/cnn_lstm_3h_quality \
  --parquet data/processed/bdg2_cleaned_3h.parquet \
  --building_split data/building_split_top_n.json \
  --output predictions.csv
```

#### 7. Monitor Training (TensorBoard)
```bash
docker compose run --rm -p 6006:6006 trainer-gpu tensorboard --logdir logs --host 0.0.0.0
# Access at http://localhost:6006
```

### Interactive Dashboard (Optional)
```bash
docker compose --profile dashboard up dash
# Access at http://localhost:8501
```

## Features

### Data Processing
- **Automated ingestion** from BDG2 GitHub repository
- **GPU-accelerated preprocessing** with RAPIDS (optional, CPU fallback)
- **Data quality analysis** with building selection and train/val/test split
- **3-hour resampling** for memory-efficient training on limited VRAM
- **Memory-efficient**: Lazy loading with TensorFlow generators

### Model Capabilities
- **Hybrid CNN-LSTM**: CNN for local features + LSTM for temporal dependencies
- **Bidirectional LSTM**: Forward and backward context for better predictions
- **Mixed-precision training**: FP16 for 2x speedup
- **Batch inference**: Predict on test buildings with per-building metrics
- **Baseline comparisons**: Persistence, seasonal, and linear regression baselines

### Production-Ready
- **Containerized workflow**: Full Docker setup with GPU support
- **Configuration management**: Dataclass-based config with JSON serialization
- **Reproducible training**: Fixed random seeds and deterministic operations
- **Comprehensive logging**: TensorBoard integration
- **CI/CD**: GitHub Actions workflow with pre-commit hooks (ruff, mypy)
- **Testing**: pytest suite with unit and integration tests

## Development

### Dependencies
Dependencies are managed via `pyproject.toml`:
```bash
# Install base dependencies
pip install .

# Install dev dependencies (pytest, ruff, mypy, pre-commit)
pip install -e ".[dev]"

# Install GPU preprocessing (optional)
pip install ".[gpu]"
```

### Running Tests
```bash
pytest                       # Run all tests
pytest -m "not gpu"          # Skip GPU-dependent tests
pytest --cov=src             # With coverage
```

### Pre-commit Hooks
```bash
pre-commit install           # Install hooks
pre-commit run --all-files   # Run manually
```

## Troubleshooting

### LSTM cuDNN Warnings (Expected)
```
WARNING: Layer lstm_0 will not use cuDNN kernels since it doesn't meet the criteria.
```
This is intentional for cuDNN 9.0+ compatibility. The warnings can be ignored.

### GPU Memory Issues
```bash
# Use 3h resampled data (recommended)
# Reduce batch size
python -m src.main --batch_size 4
```

### TensorFlow Not Using GPU
```bash
docker compose run --rm trainer-gpu python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### RAPIDS Installation Fails
RAPIDS is optional (only for preprocessing). The pipeline works without it using pandas/numpy.

## Performance Benchmarks

**Typical training time (full dataset, RTX 3060, 3h resampled data):**
- Data ingestion: ~2 min
- Preprocessing: ~1 min
- CNN-LSTM training (50 epochs): ~35-45 min
- **Total pipeline: ~40-50 min**

**Memory usage:**
- GPU VRAM: ~4-5 GB (with mixed precision)
- System RAM: ~6-8 GB (during preprocessing)

## Dataset

**Building Data Genome Project 2 (BDG2)**
- **Size**: 27.7M hourly records (~9.2M with 3h resampling)
- **Buildings**: 1,578 from 19 sites (6 countries)
- **Period**: 2016-2017 (1-2 years per building)
- **Meter**: Electricity consumption only

**GitHub**: https://github.com/buds-lab/building-data-genome-project-2

## Technology Stack

- **Deep Learning**: TensorFlow 2.15 (GPU)
- **Data Processing**: pandas, numpy, pyarrow
- **Preprocessing (optional)**: RAPIDS (cuDF, CuPy)
- **Visualization**: matplotlib, Streamlit
- **Quality**: ruff, mypy, pre-commit, pytest
- **Build**: pyproject.toml (PEP 621)
- **CI/CD**: GitHub Actions
- **Container**: Docker + NVIDIA Container Toolkit
