"""Expose the FastAPI application for deployment."""

import os

import uvicorn

from reranker_service.api import create_app

app = create_app()

__all__ = ["app"]


def main() -> None:
    """Run the FastAPI app with Uvicorn."""
    host = os.getenv("RERANKER_HOST", "0.0.0.0")
    port = int(os.getenv("RERANKER_PORT", "10000"))
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
