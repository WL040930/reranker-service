"""Expose the FastAPI application for deployment."""

import os

import uvicorn

from reranker_service.api import create_app

app = create_app()

__all__ = ["app"]


def main() -> None:
    """Run the FastAPI app with Uvicorn."""
    host = os.getenv("RERANKER_HOST", "0.0.0.0")
    port = int(os.getenv("RERANKER_PORT", "8000"))
    
    # Memory-efficient Uvicorn configuration
    uvicorn.run(
        app, 
        host=host, 
        port=port, 
        reload=False,
        workers=1,  # Single worker to minimize memory usage
        limit_concurrency=10,  # Limit concurrent connections
        timeout_keep_alive=5,  # Reduce keep-alive timeout
        access_log=False  # Disable access logs to save memory
    )


if __name__ == "__main__":
    main()
