"""Cross-encoder powered reranking service."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from cachetools import TTLCache
from sentence_transformers import CrossEncoder

from .config import AppConfig, get_config

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Service object responsible for reranking documents with a cross-encoder model."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or get_config()
        self._cache: TTLCache[str, List[Dict[str, Any]]] = TTLCache(
            maxsize=self.config.cache_size,
            ttl=self.config.cache_ttl_seconds,
        )
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._model = self._load_model()

    def _load_model(self) -> CrossEncoder:
        """Load the configured cross-encoder model."""
        logger.info("Loading cross-encoder model %s", self.config.model_name)
        model = CrossEncoder(self.config.model_name, max_length=self.config.max_length)
        logger.info("Cross-encoder model loaded")
        return model

    @staticmethod
    def _normalize_documents(documents: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure documents are dictionaries with text fields available."""
        normalized = []
        for index, doc in enumerate(documents):
            if isinstance(doc, dict):
                text = doc.get("text") or doc.get("content") or doc.get("answer") or ""
                normalized.append(
                    {
                        "index": index,
                        "text": text,
                        "metadata": doc.get("metadata") or {},
                        "raw": doc,
                    }
                )
            else:
                normalized.append(
                    {
                        "index": index,
                        "text": str(doc),
                        "metadata": {},
                        "raw": {"text": str(doc)},
                    }
                )
        return normalized

    @staticmethod
    def _cache_key(query: str, documents: Iterable[Dict[str, Any]]) -> str:
        """Generate a deterministic cache key for a query + documents request."""
        digest = hashlib.sha256()
        digest.update(query.encode("utf-8"))
        for doc in documents:
            digest.update(str(doc.get("text", "")).encode("utf-8"))
        return digest.hexdigest()[:32]

    def _model_inputs(self, query: str, documents: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """Prepare model input pairs."""
        return [(query, doc["text"]) for doc in documents]

    async def rerank(
        self,
        query: str,
        documents: Sequence[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return reranked documents sorted by cross-encoder score."""
        logger.debug("Received rerank request for %d documents", len(documents))

        if not query or not documents:
            return []

        normalized = self._normalize_documents(documents)
        cache_key = self._cache_key(query, normalized)
        if cache_key in self._cache:
            logger.debug("Returning cached rerank result")
            cached = self._cache[cache_key]
            return cached[:top_k] if top_k is not None else list(cached)

        pairs = self._model_inputs(query, normalized)

        loop = asyncio.get_running_loop()
        scores: List[float] = await loop.run_in_executor(
            self._executor,
            lambda: self._model.predict(pairs),
        )

        ranked = []
        for doc, score in zip(normalized, scores):
            ranked.append(
                {
                    "index": doc["index"],
                    "score": float(score),
                    "document": doc["raw"],
                }
            )

        ranked.sort(key=lambda item: item["score"], reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        self._cache[cache_key] = list(ranked)
        return ranked

    async def health(self) -> bool:
        """Return True when the model is available for inference."""
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor,
                lambda: self._model.tokenizer is not None,
            )
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Health check failed: %s", exc)
            return False
