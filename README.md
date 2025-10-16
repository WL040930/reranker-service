## Reranker Service

FastAPI application that loads the `cross-encoder/ms-marco-MiniLM-L-12-v2`
model from `sentence-transformers` and exposes HTTP endpoints to rerank documents
for a query.

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

- `GET /health` returns `{"status":"ok"}` when the model is ready.

### Run locally

```bash
uvicorn reranker_service.api:app --host 0.0.0.0 --port 8000
```

Model name and cache behaviour can be configured through environment variables:

- `RERANKER_MODEL_NAME`
- `RERANKER_MAX_LENGTH`
- `RERANKER_CACHE_SIZE`
- `RERANKER_CACHE_TTL_SECONDS`
- `RERANKER_REQUEST_TIMEOUT_SECONDS`
- `RERANKER_LOG_LEVEL`
