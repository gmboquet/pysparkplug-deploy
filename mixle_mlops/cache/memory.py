"""In-process, thread-safe TTL cache. The default backend when no Redis URL is configured: good for a single
gateway process and for tests. Not shared across replicas — use ``RedisCache`` for that."""
from __future__ import annotations

import threading
import time
from typing import Any

from .base import Cache


class MemoryCache(Cache):
    """A dict-backed cache with per-key expiry, guarded by a lock so it is safe under many concurrent
    request handlers. Expired entries are dropped lazily on access and opportunistically on writes."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[Any, float | None]] = {}   # key -> (value, expires_at | None)
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _expired(self, expires_at: float | None, now: float) -> bool:
        return expires_at is not None and expires_at <= now

    def _purge(self, now: float) -> None:
        dead = [k for k, (_, exp) in self._data.items() if self._expired(exp, now)]
        for k in dead:
            self._data.pop(k, None)

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None
            value, expires_at = entry
            if self._expired(expires_at, now):
                self._data.pop(key, None)
                self._misses += 1
                return None
            self._hits += 1
            return value

    def set(self, key: str, value: Any, *, ttl: float | None = None) -> None:
        now = time.monotonic()
        expires_at = now + ttl if ttl is not None else None
        with self._lock:
            self._data[key] = (value, expires_at)
            # cheap opportunistic purge so the dict doesn't grow unbounded with expired keys
            if len(self._data) % 256 == 0:
                self._purge(now)

    def incr(self, key: str, amount: int = 1, *, ttl: float | None = None) -> int:
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None or self._expired(entry[1], now):
                expires_at = now + ttl if ttl is not None else None
                new = int(amount)
                self._data[key] = (new, expires_at)
                return new
            value, expires_at = entry
            new = int(value) + int(amount)
            self._data[key] = (new, expires_at)
            return new

    def expire(self, key: str, ttl: float) -> bool:
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None or self._expired(entry[1], now):
                return False
            self._data[key] = (entry[0], now + ttl)
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._data.pop(key, None) is not None

    def stats(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            self._purge(now)
            total = self._hits + self._misses
            return {
                "backend": "memory",
                "size": len(self._data),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (self._hits / total) if total else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0
