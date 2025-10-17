"""Expose the FastAPI application for deployment."""

import os
import logging
import gc

import uvicorn

# Set memory optimization environment variables before any imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Suppress progress bars and verbose logging from transformers
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Disable tqdm progress bars globally
try:
    import tqdm
    tqdm.tqdm.disable = True
except ImportError:
    pass

# Memory-optimized default configuration
os.environ.setdefault("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-TinyBERT-L-2-v2")
os.environ.setdefault("RERANKER_CACHE_SIZE", "4")
os.environ.setdefault("RERANKER_MAX_LENGTH", "64")
os.environ.setdefault("RERANKER_PRELOAD_MODEL", "true")  # Enable preloading by default

from src.api import create_app

# Configure aggressive garbage collection for memory optimization
gc.set_threshold(100, 5, 5)

# Create the app instance
app = create_app()

__all__ = ["app"]


def main() -> None:
    """Run the FastAPI app with memory-optimized Uvicorn configuration."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    host = os.getenv("RERANKER_HOST", "0.0.0.0")
    port = int(os.getenv("RERANKER_PORT", "8000"))
    
    logger.info("🚀 Starting Memory-Optimized Reranker Service")
    logger.info(f"   Host: {host}:{port}")
    logger.info(f"   Model: {os.getenv('RERANKER_MODEL_NAME')}")
    logger.info(f"   Max Length: {os.getenv('RERANKER_MAX_LENGTH')}")
    logger.info(f"   Cache Size: {os.getenv('RERANKER_CACHE_SIZE')}")
    logger.info(f"   Preload Model: {os.getenv('RERANKER_PRELOAD_MODEL')}")
    logger.info("   Memory Target: <400MB")
    
    try:
        # Performance-optimized Uvicorn configuration
        uvicorn_config = {
            "app": app,
            "host": host,
            "port": port,
            "reload": False,
            "workers": 4,                    # Multiple workers for better throughput
            "limit_concurrency": 50,         # Higher concurrent connections
            "timeout_keep_alive": 30,        # Longer keep-alive timeout
            "access_log": True,              # Enable access logs
            "server_header": True,           # Enable server header
            "date_header": True,             # Enable date header
            "log_level": "info",             # Standard logging
        }
        
        # Try to use uvloop if available, fallback to asyncio
        try:
            import uvloop
            uvicorn_config["loop"] = "uvloop"
            logger.info("Using uvloop for enhanced performance")
        except ImportError:
            logger.info("uvloop not available, using asyncio")
            uvicorn_config["loop"] = "asyncio"
        
        # Try to use httptools if available, fallback to h11
        try:
            import httptools
            uvicorn_config["http"] = "httptools"
            logger.info("Using httptools for enhanced HTTP performance")
        except ImportError:
            logger.info("httptools not available, using h11")
            uvicorn_config["http"] = "h11"
        
        uvicorn.run(**uvicorn_config)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


if __name__ == "__main__":
    main()
