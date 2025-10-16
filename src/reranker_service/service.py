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

    def __init__(self, config: Optional[AppConfig] = None, preload_model: bool = False) -> None:
        self.config = config or get_config()
        # Use smaller cache to save memory
        self._cache: TTLCache[str, List[Dict[str, Any]]] = TTLCache(
            maxsize=max(1, self.config.cache_size // 4),  # Quarter the cache size
            ttl=self.config.cache_ttl_seconds,
        )
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._model: Optional[CrossEncoder] = None
        self._model_loaded = False
        
        # Preload model if requested (for memory-optimized startup)
        if preload_model:
            self._load_model()

    def _load_model(self) -> CrossEncoder:
        """Load the configured cross-encoder model with memory optimizations."""
        if self._model is not None:
            return self._model
            
        logger.info("Loading cross-encoder model %s with memory optimizations", self.config.model_name)
        
        # Import torch here to avoid memory allocation on startup
        import torch
        
        # Configure PyTorch for minimal memory usage
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_grad_enabled(False)
        
        try:
            # Suppress progress bars for cleaner output
            import os
            os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
            
            # Suppress sentence-transformers progress bars
            import logging
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            
            # Load model with optimized configuration
            model = CrossEncoder(
                self.config.model_name, 
                max_length=self.config.max_length,
                device='cpu',
                trust_remote_code=False
            )
            
            # Optimize model for inference
            model.model.eval()
            
            # Disable gradient computation for all parameters
            for param in model.model.parameters():
                param.requires_grad = False
            
            self._model = model
            self._model_loaded = True
            
            # Test the model with a simple validation
            try:
                test_pairs = [
                    ("machine learning", "Machine learning is a method of data analysis"),
                    ("machine learning", "This is about cooking and recipes")
                ]
                test_scores = model.predict(test_pairs)
                
                # Handle potential NaN scores gracefully
                valid_scores = [s for s in test_scores if math.isfinite(float(s))]
                if len(valid_scores) == len(test_scores):
                    logger.info(f"Model validation successful. Test scores: {test_scores}")
                else:
                    logger.warning(f"Some test scores were invalid: {test_scores}")
                    
            except Exception as e:
                logger.warning(f"Model validation failed: {e}")
            
            # Force garbage collection
            gc.collect()
            
            # Log memory usage if psutil is available
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                logger.info(f"Memory usage after model load: {memory_mb:.1f} MB")
                
                if memory_mb > 500:
                    logger.warning("Memory usage exceeds 500 MB - consider using a smaller model")
                elif memory_mb > 400:
                    logger.warning("Memory usage is high (>400 MB)")
                else:
                    logger.info("Memory usage is within acceptable limits")
                    
            except ImportError:
                logger.debug("psutil not available for memory monitoring")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_name}: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

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
        logger.debug(f"Reranking query: '{query[:100]}...' with {len(pairs)} document pairs")
        for i, (q, doc_text) in enumerate(pairs[:3]):  # Log first 3 pairs
            logger.debug(f"Pair {i}: query_len={len(q)}, doc_len={len(doc_text)}")

        def predict_with_error_handling(pairs):
            """Predict with better error handling."""
            try:
                return model.predict(pairs)
            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
                # Return zeros for all pairs as fallback
                return [0.0] * len(pairs)

        loop = asyncio.get_running_loop()
        scores: List[float] = await loop.run_in_executor(
            self._executor,
            predict_with_error_handling,
            pairs,
        )
        
        # Debug: Log the raw scores
        logger.debug(f"Raw scores from model: {scores[:10]}")  # Log first 10 scores

        ranked = []
        for doc, score in zip(normalized, scores):
            # Handle NaN/infinite scores gracefully
            try:
                if hasattr(score, 'item'):  # NumPy scalar
                    safe_score = float(score.item())
                else:
                    safe_score = float(score)
                    
                if not math.isfinite(safe_score):
                    logger.debug(f"Non-finite score for doc {doc['index']}: {score}, using 0.0")
                    safe_score = 0.0
                    
            except (ValueError, TypeError, AttributeError):
                logger.debug(f"Invalid score type for doc {doc['index']}: {type(score)}, using 0.0")
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
