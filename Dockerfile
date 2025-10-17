# Use Python 3.11 slim image for smaller size and better performance
FROM python:3.11-slim

# Set environment variables for Python optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Performance optimization environment variables (allowing more resources)
ENV TOKENIZERS_PARALLELISM=true \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    TRANSFORMERS_VERBOSITY=warning \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# ====================================
# APPLICATION ENVIRONMENT VARIABLES
# ====================================
# These environment variables control the reranker service behavior.
# Override any of these when running the container with -e flag.

# Model Configuration:
# RERANKER_MODEL_NAME: HuggingFace model name for cross-encoder reranking
#   Default: cross-encoder/ms-marco-MiniLM-L-6-v2 (better performance model)
#   Memory-optimized alternative: cross-encoder/ms-marco-TinyBERT-L-2-v2
ENV RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2

# RERANKER_MAX_LENGTH: Maximum token length for input sequences
#   Default: 256 (higher for better accuracy)
#   Memory-optimized: 64
ENV RERANKER_MAX_LENGTH=256

# RERANKER_CACHE_SIZE: Number of queries to cache in memory
#   Default: 32 (higher for better performance)
#   Memory-optimized: 4
ENV RERANKER_CACHE_SIZE=32

# RERANKER_CACHE_TTL_SECONDS: Cache time-to-live in seconds
#   Default: 600 (10 minutes)
ENV RERANKER_CACHE_TTL_SECONDS=600

# RERANKER_REQUEST_TIMEOUT_SECONDS: Request timeout in seconds
#   Default: 60.0
ENV RERANKER_REQUEST_TIMEOUT_SECONDS=60.0

# RERANKER_PRELOAD_MODEL: Whether to preload model at startup
#   Default: true (recommended for production)
ENV RERANKER_PRELOAD_MODEL=true

# RERANKER_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
#   Default: INFO
ENV RERANKER_LOG_LEVEL=INFO

# Server Configuration:
# RERANKER_HOST: Host to bind the server to
#   Default: 0.0.0.0 (all interfaces)
ENV RERANKER_HOST=0.0.0.0

# RERANKER_PORT: Port to bind the server to
#   Default: 8000
ENV RERANKER_PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml ./

# Upgrade pip and install build dependencies first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -e . --verbose

# Copy application code
COPY src/ ./src/
COPY app.py ./

# Change ownership to app user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Create cache directory for models
RUN mkdir -p /home/app/.cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "app.py"]