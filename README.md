## Reranker Service

FastAPI application that loads the `cross-encoder/ms-marco-MiniLM-L-6-v2`
model from `sentence-transformers` and exposes HTTP endpoints to rerank documents
for a query.

**Memory Optimized:** This service is designed to run efficiently on servers with 512MB RAM or less by using lazy model loading, reduced cache sizes, and CPU-only inference.

### Endpoints

- `POST /rerank`  
  Request body:
  ```json
  {
    "query": "who founded fastapi",
    "documents": [
      {"text": "FastAPI was created by Sebastian Ramirez."},
      {"text": "Flask is a Python microframework released in 2010."}
    ],
    "top_k": 1
  }
  ```
  Response:
  ```json
  {
    "rankings": [{"index": 0, "score": 0.91}],
    "top_k_applied": true
  }
  ```

- `GET /health` returns `{"status":"ok"}` when the service is ready (lightweight check that doesn't load the model).

- `GET /metrics` returns memory usage and performance metrics:
  ```json
  {
    "memory": {
      "rss_mb": 85.4,
      "vms_mb": 512.3
    },
    "cpu_percent": 2.5,
    "num_threads": 4
  }
  ```

### Run locally

#### Option 1: Using environment files (recommended)

```bash
# For 512MB servers (development/small servers)
cp .env.dev .env
python -m uvicorn reranker_service.api:app --host 0.0.0.0 --port 8000

# For production (1GB+ servers)
cp .env.production .env
python -m uvicorn reranker_service.api:app --host 0.0.0.0 --port 8000

# For very tight memory constraints
cp .env.minimal .env
python -m uvicorn reranker_service.api:app --host 0.0.0.0 --port 8000
```

#### Option 2: Direct command

```bash
uvicorn reranker_service.api:app --host 0.0.0.0 --port 8000
```

### Memory Optimization

The service uses several strategies to minimize memory usage:

1. **Lazy Model Loading**: The ML model is only loaded when the first `/rerank` request is made, not during startup or health checks
2. **CPU-Only Mode**: Uses PyTorch CPU backend to avoid GPU memory allocation
3. **Reduced Cache**: Default cache size reduced to 32 entries (configurable)
4. **Smaller Max Length**: Default token length reduced to 256 (configurable)
5. **Single Worker**: Uvicorn runs with a single worker process
6. **Gradient Disabled**: Model runs in inference-only mode with no gradient computation

### Configuration

Model name and cache behaviour can be configured through environment variables:

- `RERANKER_MODEL_NAME` (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `RERANKER_MAX_LENGTH` (default: `256` - reduced for memory efficiency)
- `RERANKER_CACHE_SIZE` (default: `32` - reduced for memory efficiency)
- `RERANKER_CACHE_TTL_SECONDS` (default: `300`)
- `RERANKER_REQUEST_TIMEOUT_SECONDS` (default: `30.0`)
- `RERANKER_LOG_LEVEL` (default: `INFO`)

### Memory Usage Tips

For 512MB servers:
- The health endpoint is now lightweight and won't trigger model loading
- First `/rerank` request will be slower as it loads the model
- Consider using swap space if needed for model loading
- Monitor memory with: `docker stats` or `free -h`
