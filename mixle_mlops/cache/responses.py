"""Response caching for chat completions.

Two layers, both over a ``Cache`` backend:

  * ``ResponseCache`` — *exact* match. Keyed by ``hash(model + messages + sampling params)`` via
    ``chat_request_key``; an identical request returns the stored completion verbatim. This is the cheap,
    high-precision layer that absorbs duplicate traffic from many simultaneous users.

  * ``SemanticCache`` — *near-duplicate* match. The integrator INJECTS an embedder callable (the RAG
    subpackage's embedder); this module never imports ``rag``. We embed the request's final user turn, store
    ``(embedding, response)`` per model, and on lookup return a cached answer when cosine-similarity to a
    stored entry exceeds ``threshold``. Bounded by ``max_entries`` (LRU-ish FIFO eviction).

Both layers degrade gracefully: a missing embedder or empty store simply means "miss".
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Sequence

from .base import Cache, chat_request_key, cache_key

Embedder = Callable[[str], Sequence[float]]


def _last_user_text(messages: Any) -> str:
    """The text of the final user turn — what the semantic cache matches on."""
    for m in reversed(list(messages or [])):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role != "user":
            continue
        if isinstance(m, dict):
            content = m.get("content", "")
        elif hasattr(m, "text"):
            return m.text()
        else:
            content = getattr(m, "content", "")
        if isinstance(content, str):
            return content
        # list of parts → concat text parts
        parts = []
        for p in content or []:
            t = p.get("text") if isinstance(p, dict) else getattr(p, "text", None)
            if t:
                parts.append(t)
        return "".join(parts)
    return ""


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return -1.0
    return dot / (na * nb)


class ResponseCache:
    """Exact chat-completion cache over a ``Cache`` backend."""

    def __init__(self, cache: Cache, *, ttl: float | None = 3600.0, prefix: str = "resp:exact:"):
        self.cache = cache
        self.ttl = ttl
        self.prefix = prefix

    def key_for(self, req: Any) -> str:
        """Build the exact key from a ``ChatRequest``-like object (pydantic or dict)."""
        if isinstance(req, dict):
            model = req.get("model", "")
            messages = req.get("messages", [])
            temperature = req.get("temperature")
            max_tokens = req.get("max_tokens")
            top_p = req.get("top_p")
            extra = req.get("extra")
        else:
            model = getattr(req, "model", "")
            messages = getattr(req, "messages", [])
            temperature = getattr(req, "temperature", None)
            max_tokens = getattr(req, "max_tokens", None)
            top_p = getattr(req, "top_p", None)
            extra = getattr(req, "extra", None)
        return chat_request_key(
            model, messages, temperature=temperature, max_tokens=max_tokens,
            top_p=top_p, extra=extra, prefix=self.prefix,
        )

    def get(self, req: Any) -> Any | None:
        return self.cache.get(self.key_for(req))

    def set(self, req: Any, response: Any) -> None:
        self.cache.set(self.key_for(req), response, ttl=self.ttl)


class SemanticCache:
    """Near-duplicate chat-completion cache. Embeds the final user turn with an INJECTED embedder and returns
    a cached answer when cosine-similarity to a stored entry beats ``threshold``.

    Storage: one index list per model under ``{prefix}{model}`` in the backing ``Cache`` (each entry is
    ``{"emb": [...], "response": ..., "ts": ...}``). The embedder is supplied by the integrator (RAG's) so
    this module has no RAG dependency."""

    def __init__(
        self,
        cache: Cache,
        embedder: Embedder | None = None,
        *,
        threshold: float = 0.9,
        ttl: float | None = 3600.0,
        max_entries: int = 256,
        prefix: str = "resp:sem:",
    ):
        self.cache = cache
        self.embedder = embedder
        self.threshold = threshold
        self.ttl = ttl
        self.max_entries = max_entries
        self.prefix = prefix

    def _index_key(self, model: str) -> str:
        return cache_key({"model": model or ""}, prefix=self.prefix)

    def _embed(self, text: str) -> list[float] | None:
        if self.embedder is None or not text:
            return None
        try:
            vec = self.embedder(text)
        except Exception:
            return None
        return [float(x) for x in vec] if vec is not None else None

    def lookup(self, req: Any) -> tuple[Any | None, float]:
        """Return ``(response, similarity)`` for the best near-duplicate, or ``(None, score)`` on a miss."""
        model, messages = self._model_messages(req)
        emb = self._embed(_last_user_text(messages))
        if emb is None:
            return None, -1.0
        index = self.cache.get(self._index_key(model)) or []
        best_resp, best_sim = None, -1.0
        for entry in index:
            sim = _cosine(emb, entry.get("emb", []))
            if sim > best_sim:
                best_sim, best_resp = sim, entry.get("response")
        if best_sim >= self.threshold:
            return best_resp, best_sim
        return None, best_sim

    def store(self, req: Any, response: Any) -> bool:
        """Index ``response`` under the request's embedded user turn. Returns ``False`` if no embedder/text."""
        model, messages = self._model_messages(req)
        emb = self._embed(_last_user_text(messages))
        if emb is None:
            return False
        key = self._index_key(model)
        index = self.cache.get(key) or []
        index.append({"emb": emb, "response": response, "ts": time.time()})
        if len(index) > self.max_entries:        # FIFO eviction of the oldest entries
            index = index[-self.max_entries:]
        self.cache.set(key, index, ttl=self.ttl)
        return True

    @staticmethod
    def _model_messages(req: Any) -> tuple[str, Any]:
        if isinstance(req, dict):
            return req.get("model", ""), req.get("messages", [])
        return getattr(req, "model", ""), getattr(req, "messages", [])
