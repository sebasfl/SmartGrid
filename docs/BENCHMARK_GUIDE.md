# Benchmarking Guide

Esta guía te ayudará a dimensionar el tiempo de entrenamiento y los recursos necesarios para entrenar el modelo MTL Transformer en tu sistema.

## ¿Por qué hacer benchmarking?

Antes de ejecutar entrenamientos largos (50-200 épocas), es importante saber:
- ⏱️ **Cuánto tiempo** tomará el entrenamiento completo
- 💻 **Cuántos recursos** consumirá (GPU, RAM, CPU)
- ⚙️ **Qué configuración** (batch_size) es óptima para tu hardware
- 🚨 **Si tu sistema** puede manejar el dataset completo

## Uso Rápido

### 1. Benchmark Básico (Recomendado para empezar)

```bash
docker compose run --rm trainer-gpu python -m src.benchmark \
  --parquet data/processed/bdg2_cleaned.parquet \
  --output_dir data/benchmark
```

**Qué hace:**
- Ejecuta 3 épocas de entrenamiento
- Usa dataset limitado (150 buildings de entrenamiento)
- Monitorea uso de CPU, RAM, GPU
- Genera reporte con estimaciones de tiempo

**Tiempo estimado:** 5-15 minutos dependiendo de tu GPU

### 2. Benchmark con Dataset Completo

```bash
docker compose run --rm trainer-gpu python -m src.benchmark \
  --parquet data/processed/bdg2_cleaned.parquet \
  --output_dir data/benchmark \
  --use_full_dataset
```

**Qué hace:**
- Ejecuta 3 épocas con TODOS los buildings (1,578)
- Requiere al menos 6GB de RAM
- Proporciona estimaciones más precisas para producción

**Tiempo estimado:** 15-30 minutos dependiendo de tu GPU

### 3. Benchmark Personalizado

```bash
docker compose run --rm trainer-gpu python -m src.benchmark \
  --parquet data/processed/bdg2_cleaned.parquet \
  --benchmark_epochs 5 \
  --batch_size 64 \
  --output_dir data/benchmark \
  --use_full_dataset
```

**Parámetros:**
- `--benchmark_epochs`: Más épocas = estimaciones más precisas (pero más tiempo)
- `--batch_size`: Prueba diferentes tamaños para encontrar el óptimo
- `--config`: Usa un archivo de configuración personalizado

## Interpretando los Resultados

### Reporte de Consola

Al finalizar, verás un reporte como este:

```
================================================================================
                         BENCHMARK REPORT
================================================================================

📊 SYSTEM INFORMATION
--------------------------------------------------------------------------------
  TensorFlow Version:    2.15.0
  CUDA Available:        True
  GPU Count:             1
  GPU Model:             NVIDIA GeForce RTX 3060
  GPU Memory:            12288 MB

⚙️  TRAINING CONFIGURATION
--------------------------------------------------------------------------------
  Batch Size:            32
  Benchmark Epochs:      3
  Mixed Precision:       True
  Dataset Mode:          full

💻 RESOURCE USAGE
--------------------------------------------------------------------------------
  CPU Usage:             avg=45.2%, max=78.1%
  Memory Usage:          avg=52.3%, max=68.7%
  GPU Usage:             avg=92.5%, max=98.2%
  GPU Memory:            avg=75.3%, max=85.1%
  GPU Temperature:       avg=68.5°C, max=72.0°C

⏱️  TIMING STATISTICS
--------------------------------------------------------------------------------
  Avg Epoch Time:        0.90 min (54.2s)
  Std Epoch Time:        ±2.3s
  Avg Batch Time:        0.125s

📈 TIME ESTIMATES
--------------------------------------------------------------------------------

  For 50 epochs:
    Full training:       0.75 hours (45.0 min)
    With early stop:     0.53 hours (31.5 min) [~35 epochs]

  For 100 epochs:
    Full training:       1.50 hours (90.0 min)
    With early stop:     1.05 hours (63.0 min) [~70 epochs]

  For 200 epochs:
    Full training:       3.00 hours (180.0 min)
    With early stop:     2.10 hours (126.0 min) [~140 epochs]

💡 RECOMMENDATIONS
--------------------------------------------------------------------------------

  1. GPU memory usage is optimal (75.3%)
     → Keep current batch_size=32

================================================================================
```

### Archivo JSON

Además del reporte de consola, se guarda un archivo JSON con todos los detalles:

```bash
data/benchmark/benchmark_results_<timestamp>.json
```

Este archivo contiene:
- Tiempos de cada época individual
- Estadísticas detalladas de cada batch
- Uso de recursos a lo largo del tiempo
- Recomendaciones y configuración usada

## Recomendaciones Según Resultados

### Si GPU Memory < 50%

```
💡 Low GPU memory usage (<50%)
   → Increase batch_size to 64 for faster training
```

**Acción:** Aumenta el batch_size para acelerar el entrenamiento:

```bash
# Prueba con batch_size mayor
docker compose run --rm trainer-gpu python -m src.benchmark \
  --parquet data/processed/bdg2_cleaned.parquet \
  --batch_size 64
```

### Si GPU Memory > 90%

```
💡 High GPU memory usage (>90%)
   → Decrease batch_size to 16 to avoid OOM errors
```

**Acción:** Reduce el batch_size para evitar errores de memoria:

```bash
# Prueba con batch_size menor
docker compose run --rm trainer-gpu python -m src.benchmark \
  --parquet data/processed/bdg2_cleaned.parquet \
  --batch_size 16
```

### Si GPU Memory 50-90%

```
💡 GPU memory usage is optimal (75%)
   → Keep current batch_size=32
```

**Acción:** Tu configuración es óptima, procede con el entrenamiento.

## Ejemplos de Hardware

### RTX 3060 (12GB VRAM)

```
Dataset completo:
  Batch Size:     32
  Época:          ~54 segundos
  50 épocas:      ~45 minutos (31 min con early stop)
  GPU Memory:     ~75%
  Recomendación:  Óptimo, usar batch_size=32
```

### RTX 3050 (8GB VRAM)

```
Dataset completo:
  Batch Size:     16 (reducido para evitar OOM)
  Época:          ~78 segundos
  50 épocas:      ~65 minutos (45 min con early stop)
  GPU Memory:     ~85%
  Recomendación:  Usar batch_size=16, considerar dataset limitado
```

### GTX 1660 Ti (6GB VRAM)

```
Dataset limitado (recomendado):
  Batch Size:     16
  Época:          ~45 segundos
  50 épocas:      ~37 minutos (26 min con early stop)
  GPU Memory:     ~88%
  Recomendación:  Usar dataset limitado, batch_size=16
```

## Flujo de Trabajo Recomendado

### 1. Primera vez (sin datos históricos)

```bash
# Paso 1: Benchmark básico para conocer tu hardware
docker compose run --rm trainer-gpu python -m src.benchmark \
  --parquet data/processed/bdg2_cleaned.parquet

# Paso 2: Revisa el reporte y ajusta batch_size si es necesario

# Paso 3: Benchmark con dataset completo si tu sistema lo soporta
docker compose run --rm trainer-gpu python -m src.benchmark \
  --parquet data/processed/bdg2_cleaned.parquet \
  --use_full_dataset

# Paso 4: Usa las estimaciones para planear tu entrenamiento
```

### 2. Experimentando con configuraciones

```bash
# Prueba diferentes batch sizes
for bs in 16 32 64; do
  docker compose run --rm trainer-gpu python -m src.benchmark \
    --parquet data/processed/bdg2_cleaned.parquet \
    --batch_size $bs \
    --output_dir data/benchmark
done

# Compara los resultados en data/benchmark/*.json
```

### 3. Antes de entrenamientos largos

```bash
# Antes de 100-200 épocas, haz benchmark de 5 épocas
docker compose run --rm trainer-gpu python -m src.benchmark \
  --parquet data/processed/bdg2_cleaned.parquet \
  --benchmark_epochs 5 \
  --use_full_dataset

# Las estimaciones serán más precisas con más épocas
```

## Troubleshooting

### Error: Out of Memory (OOM)

```
ResourceExhaustedError: OOM when allocating tensor
```

**Solución:**
1. Reduce batch_size a 16 o 8
2. Usa dataset limitado (sin `--use_full_dataset`)
3. Considera usar solo CPU si tu GPU es muy pequeña

### Benchmark muy lento

Si el benchmark tarda mucho:
- Reduce `--benchmark_epochs` a 2
- Usa dataset limitado para pruebas iniciales
- Verifica que estés usando GPU (no CPU)

### No se detecta GPU

```
⚠️  No GPU detected - will run on CPU
```

**Solución:**
1. Verifica que nvidia-docker esté instalado
2. Ejecuta `nvidia-smi` para verificar que la GPU esté disponible
3. Reconstruye el contenedor: `docker compose build --no-cache trainer-gpu`

## Preguntas Frecuentes

### ¿Cuántas épocas debo usar para el benchmark?

- **2-3 épocas**: Suficiente para estimación rápida (5-15 min)
- **5 épocas**: Mejor precisión, recomendado antes de entrenamientos largos (15-30 min)
- **10 épocas**: Muy preciso, solo si tienes tiempo

### ¿Debo usar dataset completo para el benchmark?

- **Sí** si planeas entrenar con dataset completo en producción
- **No** si solo estás probando el pipeline o tu hardware es limitado (<6GB RAM)

### ¿Los resultados son exactos?

Los resultados son **estimaciones** basadas en:
- Promedio de épocas del benchmark
- Suposición de 70% de épocas completadas (early stopping)
- Condiciones actuales del sistema

Las estimaciones son generalmente precisas ±10-15%.

### ¿Puedo comparar diferentes GPUs?

Sí, ejecuta el mismo comando en diferentes máquinas y compara los archivos JSON generados.

## Siguiente Paso

Una vez que tengas las estimaciones de tu benchmark, procede con el entrenamiento completo:

```bash
# Usar la configuración óptima del benchmark
docker compose run --rm trainer-gpu python -m src.main \
  --parquet data/processed/bdg2_cleaned.parquet \
  --model_dir models/mtl_production \
  --epochs 50 \
  --batch_size 32 \
  --use_full_dataset
```

Consulta `CLAUDE.md` para más detalles sobre el entrenamiento completo.
