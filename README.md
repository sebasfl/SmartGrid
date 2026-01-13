# SmartGrid - Multi-Task Learning Transformer for Energy Forecasting

A complete **Multi-Task Learning (MTL) Transformer** pipeline for energy consumption forecasting and anomaly detection using the Building Data Genome Project 2 (BDG2) dataset. Features GPU-accelerated preprocessing, Transformer encoder architecture with dual prediction heads, and mixed-precision training.

## 🎯 Project Overview

SmartGrid implements an end-to-end deep learning pipeline that:
- **Forecasts** energy consumption 24 hours ahead (regression)
- **Detects** anomalies in historical consumption patterns (classification)
- Uses a **shared Transformer encoder** for multi-task learning
- Processes **27.7M records** from **1,578 buildings**

## 🏗️ Architecture

### MTL Transformer Model
- **Encoder**: 4-layer Transformer (d_model=128, 8 heads, 512 FFN dim)
- **Positional Encoding**: Sinusoidal (learnable option available)
- **Dual Heads**:
  - **Anomaly Detection**: Global pooling (avg+max) → MLP → Sigmoid
  - **Forecasting**: Last timestep → MLP → Dense(horizon)
- **Parameters**: ~500K trainable
- **Loss**: α=0.3 (anomaly BCE) + β=0.7 (forecast MSE)

### Training Configuration
- **Optimizer**: Adam (LR=1e-4, gradient clip=1.0)
- **Mixed Precision**: FP16 for 2x speedup
- **Callbacks**: Early stopping (patience=10), model checkpointing, LR plateau reduction
- **Data Split**: 70% train, 15% val, 15% test (by buildings)
- **Sequences**: Lookback=168h (7 days), Horizon=24h (1 day)

## 🚀 Quick Start

### Prerequisites
- Docker with NVIDIA GPU support
- NVIDIA Container Toolkit installed
- GPU with 6GB+ VRAM (GTX 1660 Ti or better)

### Complete Pipeline (Step-by-Step)

#### 1. Build GPU Container
```bash
docker compose build --no-cache trainer-gpu
```

#### 2. Data Ingestion
```bash
# Fetch BDG2 data and convert to parquet (creates data/ directory automatically)
docker compose run --rm trainer-gpu python -m src.ingest.fetch_bdg2

# Output:
#   - data/raw_electricity.csv (wide format)
#   - data/processed/bdg2_electricity_long.parquet (long format)
```

#### 3. Preprocessing (Time Features + Deduplication)
```bash
docker compose run --rm trainer-gpu python -m src.analysis.clean_building_data_parallel --parquet data/processed/bdg2_electricity_long.parquet --output data/processed/bdg2_cleaned.parquet
```

**What it does:**
- Adds 7 time features: `hour`, `day_of_week`, `month`, `is_weekend`, `is_working_hours`, `quarter`, `day_of_year`
- Removes duplicate timestamps per building
- GPU-accelerated with CuDF if available (CPU fallback)

#### 4. Train MTL Transformer

**Option A: Limited Dataset (Recommended for systems with <6GB RAM)**
```bash
# Quick test (2 epochs, 16 batch size)
docker compose run --rm trainer-gpu python -m src.main --parquet data/processed/bdg2_cleaned.parquet --model_dir models/mtl_test --epochs 2 --batch_size 16

# Full training (50 epochs, 250 train buildings)
docker compose run --rm trainer-gpu python -m src.main --parquet data/processed/bdg2_cleaned.parquet --model_dir models/mtl_limited --epochs 50
```

**Option B: Full Dataset (All 1,578 buildings - Requires 6GB+ RAM)**
```bash
docker compose run --rm trainer-gpu python -m src.main --parquet data/processed/bdg2_cleaned.parquet --model_dir models/mtl_full --epochs 50 --use_full_dataset
```

**Available CLI Arguments:**
- `--config PATH` - Path to config JSON file (optional, uses defaults if not specified)
- `--parquet PATH` - Path to preprocessed parquet file
- `--model_dir PATH` - Directory to save models
- `--epochs N` - Number of training epochs (default: 50)
- `--batch_size N` - Batch size (default: 32)
- `--use_full_dataset` - Use ALL buildings (requires 6GB+ RAM)

#### 5. Monitor Training (TensorBoard)
```bash
# In a separate terminal
docker compose run --rm -p 6006:6006 trainer-gpu tensorboard --logdir logs --host 0.0.0.0

# Access at http://localhost:6006
```

### 🎨 Interactive Dashboard (Optional)
```bash
# Start dashboard service
docker compose --profile dashboard up dash

# Access at http://localhost:8501
```

## 📊 Features

### Data Processing
- **Automated ingestion** from BDG2 GitHub repository
- **GPU-accelerated preprocessing** with RAPIDS (optional, CPU fallback)
- **Simple feature engineering**: Time-based features only
- **Memory-efficient**: Lazy loading with TensorFlow generators

### Model Capabilities
- **Multi-task learning**: Shared representations for forecasting + anomaly detection
- **Transformer architecture**: Self-attention for temporal dependencies
- **Mixed-precision training**: FP16 for 2x speedup
- **Advanced callbacks**: Early stopping, LR scheduling, checkpointing

### Production-Ready
- **Containerized workflow**: Full Docker setup with GPU support
- **Configuration management**: JSON-based hyperparameter configs
- **Reproducible training**: Fixed random seeds
- **Comprehensive logging**: TensorBoard integration

## 🔍 Troubleshooting

### GPU Memory Issues
```bash
# Reduce batch size
python -m src.main --batch_size 16

# Disable mixed precision (edit config.py)
# training.use_mixed_precision = False
```

### TensorFlow Not Using GPU
```bash
# Verify GPU is visible
docker compose run --rm trainer-gpu python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Should output: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### RAPIDS Installation Fails
RAPIDS (cuDF/CuPy) is optional and only used for preprocessing. The pipeline works without it using pandas/numpy.

## ⚡ Performance Benchmarks

**Typical training time (full dataset, RTX 3060):**
- Data ingestion: ~2 min
- Preprocessing: ~1 min (CPU) or ~30s (GPU with RAPIDS)
- MTL training (50 epochs): ~45 min
- **Total pipeline: ~50 min**

**Memory usage:**
- GPU VRAM: ~4-5 GB (with mixed precision)
- System RAM: ~8-12 GB (during preprocessing)

## 📚 Dataset

**Building Data Genome Project 2 (BDG2)**
- **Size**: 27.7M hourly records
- **Buildings**: 1,578 from 19 sites (6 countries)
- **Period**: 2016-2017 (1-2 years per building)
- **Meter**: Electricity consumption only

**GitHub**: https://github.com/buds-lab/building-data-genome-project-2

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow 2.x (GPU)
- **Data Processing**: pandas, numpy, pyarrow
- **Preprocessing (optional)**: RAPIDS (cuDF, CuPy)
- **Visualization**: matplotlib, Streamlit
- **Container**: Docker + NVIDIA Container Toolkit


