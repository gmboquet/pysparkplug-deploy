"""Build caching + concurrency primitives for the gateway: a pluggable ``Cache`` (in-process TTL or
Redis, shared across replicas), exact + semantic response caches for chat completions, and a
Cache-backed rate limiter that works across replicas.

The integrator wires the semantic cache's embedder from the RAG subpackage; this package never imports
``rag`` so it stays dependency-light and standalone."""
from __future__ import annotations

from .base import Cache, cache_key, chat_request_key
from .memory import MemoryCache
from .ratelimit import RateLimiter, RateLimitResult
from .responses import ResponseCache, SemanticCache

__all__ = [
    "Cache",
    "cache_key",
    "chat_request_key",
    "MemoryCache",
    "ResponseCache",
    "SemanticCache",
    "RateLimiter",
    "RateLimitResult",
    "get_cache",
]


def get_cache() -> Cache:
    """Process-wide cache. Picks the Redis backend when ``MIXLE_REDIS_URL`` is set (shared across gateway
    replicas), else an in-process TTL cache. Cached after first use."""
    global _cache
    if _cache is None:
        import os

        url = os.environ.get("MIXLE_REDIS_URL")
        if url:
            from .redis_cache import RedisCache

            _cache = RedisCache(url)
        else:
            _cache = MemoryCache()
    return _cache


def reset_cache() -> None:
    """Test hook: drop the cached backend so a fresh one is picked up."""
    global _cache
    _cache = None


_cache: Cache | None = None
