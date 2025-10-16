"""FastAPI application exposing the reranker service."""

from __future__ import annotations

import logging
import os
import math
import psutil
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from .config import AppConfig, get_config
from .service import CrossEncoderReranker

logger = logging.getLogger(__name__)


class Document(BaseModel):
    """Single document representation."""

    text: str
    metadata: Optional[Dict[str, Any]] = None
    
    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """Log if text is empty or problematic."""
        if not value or not value.strip():
            logger.warning("Received document with empty or whitespace-only text")
        return value


class RerankRequest(BaseModel):
    """Payload for the rerank endpoint."""

    query: str = Field(..., min_length=1)
    documents: List[Document] = Field(..., min_items=1)
    top_k: Optional[int] = Field(default=None, ge=1)

    @field_validator("documents")
    @classmethod
    def validate_documents(cls, value: List[Document]) -> List[Document]:
        if not value:
            raise ValueError("documents list must not be empty")
        return value


class Ranking(BaseModel):
    """Single rerank entry."""

    index: int
    score: float


class RerankResponse(BaseModel):
    """Response payload with rankings."""

    rankings: List[Ranking]
    top_k_applied: bool = False


@lru_cache(maxsize=1)
def get_service(config: Optional[AppConfig] = None) -> CrossEncoderReranker:
    """Return a singleton reranker service instance."""
    return CrossEncoderReranker(config=config or get_config())


def create_app() -> FastAPI:
    """Instantiate the FastAPI app."""
    config = get_config()
    app = FastAPI(
        title="Cross Encoder Reranker Service",
        version="1.0.0",
        description="HTTP API for MS-MARCO style cross-encoder reranking.",
    )

    @app.on_event("startup")
    async def configure_logging() -> None:
        logging.basicConfig(level=config.log_level)
        logger.info("Reranker service starting")

    @app.post(
        "/rerank",
        response_model=RerankResponse,
        status_code=status.HTTP_200_OK,
    )
    async def rerank(
        payload: RerankRequest,
        config: AppConfig = Depends(get_config),
    ) -> RerankResponse:
        logger.debug(
            "Rerank request received: query length=%d, documents=%d",
            len(payload.query),
            len(payload.documents),
        )
        try:
            docs = [
                {"text": doc.text, "metadata": doc.metadata or {}}
                for doc in payload.documents
            ]
            service = get_service(config=config)
            ranked = await service.rerank(
                query=payload.query,
                documents=docs,
                top_k=payload.top_k,
            )
            rankings = [
                Ranking(index=item["index"], score=item["score"])
                for item in ranked
            ]
            return RerankResponse(
                rankings=rankings,
                top_k_applied=payload.top_k is not None,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to rerank documents: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to rerank documents",
            ) from exc

    @app.get(
        "/health",
        status_code=status.HTTP_200_OK,
    )
    async def health(
        config: AppConfig = Depends(get_config),
    ) -> Dict[str, Any]:
        service = get_service(config=config)
        healthy = await service.health()
        if not healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not ready",
            )
        return {"status": "ok"}

    @app.get(
        "/metrics",
        status_code=status.HTTP_200_OK,
    )
    async def metrics() -> Dict[str, Any]:
        """Return memory usage and service metrics."""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            def _safe_float(value: float) -> float:
                try:
                    v = float(value)
                except Exception:
                    return 0.0
                if not math.isfinite(v):
                    return 0.0
                return v

            rss_mb = _safe_float(memory_info.rss / 1024 / 1024)
            vms_mb = _safe_float(memory_info.vms / 1024 / 1024)
            cpu_pct = _safe_float(process.cpu_percent(interval=0.1))
            num_threads = int(process.num_threads()) if process.num_threads() is not None else 0

            return {
                "memory": {
                    "rss_mb": round(rss_mb, 2),
                    "vms_mb": round(vms_mb, 2),
                },
                "cpu_percent": cpu_pct,
                "num_threads": num_threads,
            }
        except Exception as exc:
            logger.exception("Failed to get metrics: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve metrics",
            ) from exc

    return app


app = create_app()
