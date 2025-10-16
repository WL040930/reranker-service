"""FastAPI application exposing the reranker service."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from .config import AppConfig, get_config
from .service import CrossEncoderReranker

logger = logging.getLogger(__name__)


class Document(BaseModel):
    """Single document representation."""

    text: str
    metadata: Dict[str, Any] | None = None


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

    return app


app = create_app()
