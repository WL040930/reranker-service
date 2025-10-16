"""Application configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass
class AppConfig:
    """Runtime configuration loaded from environment variables."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    max_length: int = 512
    cache_size: int = 128
    cache_ttl_seconds: int = 900
    request_timeout_seconds: float = 30.0
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration using environment variables if provided."""

        def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
            value = os.getenv(key, default)
            if value is None:
                return None
            return value.strip() or default

        def get_int(key: str, default: int) -> int:
            raw = get_env(key)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        def get_float(key: str, default: float) -> float:
            raw = get_env(key)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        return cls(
            model_name=get_env("RERANKER_MODEL_NAME", cls.model_name),
            max_length=get_int("RERANKER_MAX_LENGTH", cls.max_length),
            cache_size=get_int("RERANKER_CACHE_SIZE", cls.cache_size),
            cache_ttl_seconds=get_int("RERANKER_CACHE_TTL_SECONDS", cls.cache_ttl_seconds),
            request_timeout_seconds=get_float(
                "RERANKER_REQUEST_TIMEOUT_SECONDS", cls.request_timeout_seconds
            ),
            log_level=get_env("RERANKER_LOG_LEVEL", cls.log_level) or cls.log_level,
        )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return cached application configuration instance."""

    return AppConfig.from_env()
