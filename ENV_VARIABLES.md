# Environment Variables Documentation

This document describes all environment variables used by the reranker service.

## Application Configuration

### Model Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace model name for cross-encoder reranking |
| `RERANKER_MAX_LENGTH` | `256` | Maximum token length for input sequences |
| `RERANKER_PRELOAD_MODEL` | `true` | Whether to preload model at startup (recommended for production) |

### Performance & Memory Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_CACHE_SIZE` | `32` | Number of queries to cache in memory |
| `RERANKER_CACHE_TTL_SECONDS` | `600` | Cache time-to-live in seconds (10 minutes) |
| `RERANKER_REQUEST_TIMEOUT_SECONDS` | `60.0` | Request timeout in seconds |

### Server Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_HOST` | `0.0.0.0` | Host to bind the server to |
| `RERANKER_PORT` | `8000` | Port to bind the server to |
| `RERANKER_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

### System Optimization

| Variable | Default | Description |
|----------|---------|-------------|
| `TOKENIZERS_PARALLELISM` | `true` | Enable parallel tokenization |
| `OMP_NUM_THREADS` | `4` | Number of OpenMP threads |
| `MKL_NUM_THREADS` | `4` | Number of MKL threads |
| `TRANSFORMERS_VERBOSITY` | `warning` | Transformers library logging level |
| `TRANSFORMERS_NO_ADVISORY_WARNINGS` | `1` | Disable advisory warnings |

## Usage Examples

### Memory-Optimized Configuration (for resource-constrained environments)
```bash
docker run -p 8000:8000 \
  -e RERANKER_MODEL_NAME=cross-encoder/ms-marco-TinyBERT-L-2-v2 \
  -e RERANKER_CACHE_SIZE=4 \
  -e RERANKER_MAX_LENGTH=64 \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  -e TOKENIZERS_PARALLELISM=false \
  reranker-service
```

### High-Performance Configuration (for powerful machines)
```bash
docker run -p 8000:8000 \
  -e RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-12-v2 \
  -e RERANKER_CACHE_SIZE=128 \
  -e RERANKER_MAX_LENGTH=512 \
  -e OMP_NUM_THREADS=8 \
  -e MKL_NUM_THREADS=8 \
  -e RERANKER_CACHE_TTL_SECONDS=3600 \
  reranker-service
```

### Development Configuration
```bash
docker run -p 8000:8000 \
  -e RERANKER_LOG_LEVEL=DEBUG \
  -e RERANKER_PRELOAD_MODEL=false \
  -e RERANKER_CACHE_SIZE=8 \
  reranker-service
```

## Model Options

The service supports any cross-encoder model from HuggingFace. Popular options:

### Memory-Optimized Models
- `cross-encoder/ms-marco-TinyBERT-L-2-v2` (smallest, ~40MB)
- `cross-encoder/ms-marco-MiniLM-L-2-v2` (small, ~120MB)

### Balanced Models (Default)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (good balance, ~400MB)
- `cross-encoder/ms-marco-MiniLM-L-4-v2` (medium, ~300MB)

### High-Performance Models
- `cross-encoder/ms-marco-MiniLM-L-12-v2` (large, ~500MB)
- `cross-encoder/ms-marco-electra-base` (very large, ~400MB)

## Resource Requirements by Configuration

| Configuration | Memory Usage | CPU Usage | Recommended For |
|--------------|--------------|-----------|-----------------|
| Memory-Optimized | ~200MB | 1 CPU | Small instances, development |
| Default (Dockerfile) | ~1GB | 2-4 CPU | Production servers |
| High-Performance | ~2GB | 4-8 CPU | High-throughput production |

## Configuration Hierarchy

Environment variables are read in this order (later overrides earlier):
1. Default values in `AppConfig` class
2. Environment variables
3. Runtime overrides (if applicable)

The `get_config()` function caches the configuration, so changes to environment variables after startup won't take effect until the service is restarted.