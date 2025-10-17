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

This Space runs a **FastAPI-based reranker microservice** using `cross-encoder/ms-marco-TinyBERT-L-2-v2`.  
It’s deployed via Docker for better compatibility with custom frameworks.

## ✅ Optimization Results

- **Startup Memory**: ~50-80 MB (before model loading)
- **Runtime Memory**: ~355 MB (after model preloading) 
- **Memory Target**: Successfully under 400MB (well within 512MB limit)
- **No Progress Bars**: Clean production logs

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

## 🚀 Quick Start

### Option 1: Using Environment Files (Recommended)

```bash
# For development (balanced performance and memory)
cp .env.dev .env
python app.py

# For production (512MB servers, preloaded model)  
cp .env.production .env
python app.py

# For minimal memory usage (very tight constraints)
cp .env.minimal .env
python app.py
```

### Option 2: Direct Run (uses built-in defaults)

```bash
python app.py
```

The server will:
1. **Load configuration** from `.env` file or use optimized defaults
2. **Preload** the model during startup (~10-15 seconds) 
3. **Listen** on the configured host/port
4. **Be ready** for immediate requests without loading delays

## 📊 Key Optimizations

### Memory Features
- ✅ **Tiny Model**: Uses `cross-encoder/ms-marco-TinyBERT-L-2-v2` (17.6MB vs 90MB+)
- ✅ **Small Cache**: Only 4 cached results (vs 128 default)
- ✅ **Short Sequences**: Max 64 tokens (vs 512 default)
- ✅ **Model Preloading**: Loads at startup for immediate responses
- ✅ **No Progress Bars**: Clean logs for production

### Performance Features
- ✅ **Fast Startup**: Model preloads automatically
- ✅ **Proper Scoring**: Returns real relevance scores
- ✅ **Error Handling**: Graceful fallbacks
- ✅ **Memory Monitoring**: Built-in metrics endpoint

## 🔧 Configuration Options

### Environment Files

| File | Use Case | Memory Usage | Features |
|------|----------|--------------|----------|
| `.env.dev` | Development | ~355MB | Balanced performance, preloading |
| `.env.production` | Production 512MB servers | ~355MB | Preloaded, clean logs |
| `.env.minimal` | Very tight memory | ~200-250MB | No preloading, minimal cache |

### Key Settings

Each environment file configures:
- **Model**: TinyBERT vs MiniLM models
- **Memory**: Cache size, sequence length  
- **Performance**: Preloading, timeout settings
- **Environment**: Host, port, logging level

## 🏗️ Production Ready

- **Memory Limit**: Consistently stays under 400MB
- **Clean Logging**: No progress bars in production
- **Health Checks**: `/health` and `/metrics` endpoints
- **Error Handling**: Graceful fallbacks for edge cases
