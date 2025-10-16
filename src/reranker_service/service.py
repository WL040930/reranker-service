"""Cross-encoder powered reranking service."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import logging
import math
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
        self._model: Optional[CrossEncoder] = None
        self._model_loaded = False

    def _load_model(self) -> CrossEncoder:
        """Load the configured cross-encoder model with memory optimizations."""
        if self._model is not None:
            return self._model
            
        logger.info("Loading cross-encoder model %s", self.config.model_name)
        
        # Force CPU-only mode to avoid GPU memory allocation
        import torch
        torch.set_num_threads(1)  # Reduce CPU threads to save memory
        
        # Fix for NaN issues: Use double precision (float64)
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        
        try:
            model = CrossEncoder(
                self.config.model_name, 
                max_length=self.config.max_length,
                device='cpu'  # Explicitly use CPU
            )
            
            # Convert model to double precision to fix NaN issues
            model.model = model.model.double()
            
            # Set model to eval mode and optimize for inference
            model.model.eval()
            
            # Disable gradient computation to save memory
            for param in model.model.parameters():
                param.requires_grad = False
            
            self._model = model
            self._model_loaded = True
            
            # Test the model with a simple prediction
            test_scores = model.predict([("test query", "test document")])
            logger.info(f"Model test prediction: {test_scores}")
            
            if not math.isfinite(float(test_scores[0])):
                logger.error("Model is producing NaN on test prediction - model may be corrupted!")
            else:
                logger.info("Model test prediction successful - no NaN values detected")
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Cross-encoder model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        finally:
            # Restore original dtype
            torch.set_default_dtype(original_dtype)

    @staticmethod
    def _normalize_documents(documents: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure documents are dictionaries with text fields available."""
        normalized = []
        for index, doc in enumerate(documents):
            if isinstance(doc, dict):
                text = doc.get("text") or doc.get("content") or doc.get("answer") or ""
                # Clean whitespace but don't filter out yet
                if isinstance(text, str):
                    text = text.strip()
                else:
                    text = str(text).strip()
                
                # Log warning but include the document
                if not text:
                    logger.warning(f"Document at index {index} has empty text - this may cause NaN scores")
                
                normalized.append(
                    {
                        "index": index,
                        "text": text,
                        "metadata": doc.get("metadata") or {},
                        "raw": doc,
                    }
                )
            else:
                text = str(doc).strip()
                if not text:
                    logger.warning(f"Document at index {index} has empty text - this may cause NaN scores")
                normalized.append(
                    {
                        "index": index,
                        "text": text,
                        "metadata": {},
                        "raw": {"text": text},
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

        # Lazy load model only when actually needed
        model = self._load_model()
        pairs = self._model_inputs(query, normalized)
        
        # Debug: Log what we're sending to the model
        logger.info(f"Reranking query: '{query[:100]}...' with {len(pairs)} document pairs")
        for i, (q, doc_text) in enumerate(pairs[:3]):  # Log first 3 pairs
            logger.info(f"Pair {i}: query_len={len(q)}, doc_len={len(doc_text)}, doc_preview='{doc_text[:100]}...'")

        loop = asyncio.get_running_loop()
        scores: List[float] = await loop.run_in_executor(
            self._executor,
            lambda: model.predict(pairs),
        )
        
        # Debug: Log the raw scores
        logger.info(f"Raw scores from model: {scores[:10]}")  # Log first 10 scores

        ranked = []
        for doc, score in zip(normalized, scores):
            # Keep the original score for now to see what's happening
            safe_score = float(score)
            if not math.isfinite(safe_score):
                logger.error(f"Model returned non-finite score: {score} for doc index {doc['index']}, text: '{doc['text'][:100]}...'")
                # Don't replace with 0.0 yet - let's see the actual issue
                safe_score = 0.0
            
            ranked.append(
                {
                    "index": doc["index"],
                    "score": safe_score,
                    "document": doc["raw"],
                }
            )

        ranked.sort(key=lambda item: item["score"], reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]

        self._cache[cache_key] = list(ranked)
        return ranked

    async def health(self) -> bool:
        """Return True when the service is ready (lightweight check)."""
        try:
            # Lightweight health check - don't load model just for health
            # Just verify the service is initialized properly
            return self.config is not None and self._executor is not None
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Health check failed: %s", exc)
            return False
