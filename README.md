---
title: Reranker Service
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# 🧠 Reranker Service (FastAPI + Docker)

FastAPI microservice that exposes a cross-encoder reranker with memory-aware defaults.
Local runs default to the lightweight `cross-encoder/ms-marco-TinyBERT-L-2-v2`, while the
Docker image is tuned for higher-accuracy `BAAI/bge-reranker-base`.

## ✨ Features
- FastAPI HTTP API with async inference and caching
- Cross-encoder reranking powered by `sentence-transformers`
- Background thread pool for CPU-bound scoring
- Built-in `/health` and `/metrics` endpoints for production monitoring
- Memory-friendly defaults (limit progress bars, single-thread execution)

## 🔌 Endpoints

- `POST /rerank` — rank documents for a given query.
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
  Sample response:
  ```json
  {
    "rankings": [{"index": 0, "score": 0.91}],
    "top_k_applied": true
  }
  ```

- `GET /health` — lightweight readiness probe (does not force model load).
- `GET /metrics` — memory (RSS/VMS), CPU usage, thread count.

## 🚀 Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

This starts the API on `0.0.0.0:7860` with the TinyBERT model, 64-token max length,
and a small cache to stay under ~400 MB once warmed.

### Test the API

```bash
curl -X POST http://localhost:7860/rerank \
  -H "Content-Type: application/json" \
  -d '{"query":"how to deploy fastapi", "documents":[{"text":"Use Uvicorn with Gunicorn."}]}'
```

## 🐳 Run with Docker

```bash
docker build -t reranker-service .
docker run --rm -p 7860:7860 reranker-service
```

The Docker image preconfigures the service for `BAAI/bge-reranker-base` (higher accuracy,
~600-800 MB RSS). Mount `/home/app/.cache` if you want to reuse downloaded models:

```bash
docker run --rm -p 7860:7860 \
  -v "$HOME/.cache/huggingface":/home/app/.cache/huggingface \
  reranker-service
```

## ⚙️ Configuration

| Variable | Default (local) | Notes |
|----------|-----------------|-------|
| `RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-TinyBERT-L-2-v2` | Docker overrides to `BAAI/bge-reranker-base`. |
| `RERANKER_MAX_LENGTH` | `64` | Docker sets `512` for BGE; reduce for memory-constrained runs. |
| `RERANKER_CACHE_SIZE` | `4` | Cache of rerank results (`cachetools.TTLCache`). |
| `RERANKER_CACHE_TTL_SECONDS` | `180` | Adjust to keep results longer. |
| `RERANKER_REQUEST_TIMEOUT_SECONDS` | `30.0` | Protect against slow scoring. |
| `RERANKER_PRELOAD_MODEL` | `true` | Skip preloading (`false`) to save startup memory. |
| `RERANKER_LOG_LEVEL` | `INFO` | Accepts any stdlib logging level. |
| `RERANKER_HOST` | `0.0.0.0` | Usually leave as-is. |
| `RERANKER_PORT` | `7860` | Change to remap without Docker. |

> Note: The `Dockerfile` exports production-friendly defaults; override any of them at runtime
with `-e KEY=value` or a `.env` file.

## 🩺 Observability & Operations
- `/health` readiness endpoint is safe for tight probes (no model load).
- `/metrics` reports RSS/VMS, CPU percent, thread count; integrate with HF Spaces or custom
  monitors.
- Logs suppress noisy progress bars and highlight model selection, memory usage, and errors.

## 🧠 Memory Tips
- Prefer the TinyBERT default when running on <512 MB RAM environments.
- Lower `RERANKER_MAX_LENGTH` or reduce the cache size for additional savings.
- Disable preloading to defer model load until the first request, trading latency for memory.
