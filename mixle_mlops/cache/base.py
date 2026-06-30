"""The ``Cache`` abstraction + cache-key helpers.

A ``Cache`` is a small string-keyed store with TTL, an atomic ``incr`` (for counters/rate limits), and a
manual ``expire``. The in-process and Redis backends implement it identically so the same response cache
and rate limiter work on a laptop or across replicas.

``cache_key`` hashes any JSON-able payload into a stable, collision-resistant hex digest; ``chat_request_key``
applies it to the request fields that actually determine the answer (model + messages + sampling params), so
two byte-different-but-semantically-identical requests collapse to one key for the exact response cache."""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any


class Cache(ABC):
    """A string-keyed store with TTL semantics. Backends: in-process (``MemoryCache``) or Redis
    (``RedisCache``, shared across replicas). Values are JSON-serialisable Python objects."""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Return the stored value, or ``None`` if absent/expired."""
        ...

    @abstractmethod
    def set(self, key: str, value: Any, *, ttl: float | None = None) -> None:
        """Store ``value`` under ``key``; expire after ``ttl`` seconds (``None`` = never)."""
        ...

    @abstractmethod
    def incr(self, key: str, amount: int = 1, *, ttl: float | None = None) -> int:
        """Atomically add ``amount`` to the integer counter at ``key`` (creating it at 0), returning the
        new value. ``ttl`` sets the expiry on first creation (used for fixed-window rate limits)."""
        ...

    @abstractmethod
    def expire(self, key: str, ttl: float) -> bool:
        """(Re)set the TTL on an existing key. Returns ``False`` if the key is absent."""
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove a key. Returns ``False`` if it was absent."""
        ...

    def stats(self) -> dict[str, Any]:
        """Best-effort backend stats for the ``/v1/cache/stats`` endpoint. Backends may override."""
        return {"backend": type(self).__name__}


def _canonical(payload: Any) -> str:
    """A stable JSON encoding: sorted keys, compact separators, so equal payloads hash equally regardless of
    dict ordering. Falls back to ``str`` for non-JSON-able leaves (e.g. pydantic objects already dumped)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def cache_key(payload: Any, *, prefix: str = "") -> str:
    """A stable, collision-resistant hex key for any JSON-able payload."""
    digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return f"{prefix}{digest}" if prefix else digest


def _messages_payload(messages: Any) -> Any:
    """Normalise messages (pydantic ``ChatMessage`` objects or plain dicts) to plain JSON-able structures so
    the key is identical whether the request arrived as objects or dicts."""
    out = []
    for m in messages or []:
        if hasattr(m, "model_dump"):
            out.append(m.model_dump())
        elif isinstance(m, dict):
            out.append(m)
        else:  # last resort
            out.append(str(m))
    return out


def chat_request_key(
    model: str,
    messages: Any,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    extra: Any = None,
    prefix: str = "chat:",
) -> str:
    """Exact-match cache key for a chat completion: hash of the fields that determine the answer. Streaming
    flag, user id, and request id are deliberately excluded so they don't fragment the cache."""
    payload = {
        "model": model or "",
        "messages": _messages_payload(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "extra": extra or {},
    }
    return cache_key(payload, prefix=prefix)
