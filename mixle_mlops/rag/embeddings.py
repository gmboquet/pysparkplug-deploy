"""Text → vector embeddings.

The primary backend is an **OpenAI-compatible** ``/v1/embeddings`` server (the same backend the LLM/embeddings/
image proxy points at — ``get_settings().llm_base_url`` / ``llm_api_key``). When no server is reachable (offline,
tests, CI) we fall back to a **deterministic local hashing embedder**: a fixed feature-hashing of character
n-grams into a unit-normalised vector. The fallback is stable across runs and processes, so identical text always
maps to an identical vector and cosine similarity is meaningful for retrieval tests — no server required.

``Embedder.embed(texts) -> np.ndarray`` of shape ``(len(texts), dim)``, L2-normalised rows.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from ..config import get_settings

# Local-fallback dimensionality. Small enough to be cheap, large enough that hashed n-gram collisions are rare.
LOCAL_DIM = 256


def _hash_embed_one(text: str, dim: int = LOCAL_DIM) -> np.ndarray:
    """Deterministic feature-hashing of character 3/4/5-grams into a unit vector.

    Signed feature hashing (sign from a second hash) keeps the representation roughly zero-mean and reduces the
    bias of pure additive collisions. Stable across processes because it only uses ``hashlib`` (no Python
    ``hash()`` randomisation).
    """
    vec = np.zeros(dim, dtype=np.float64)
    norm = (text or "").lower().strip()
    if not norm:
        return vec
    tokens = norm.split()
    grams: list[str] = list(tokens)                       # whole words as features too
    for n in (3, 4, 5):                                   # plus character n-grams (robust to typos/morphology)
        for i in range(len(norm) - n + 1):
            grams.append(norm[i : i + n])
    for g in grams:
        h = hashlib.sha1(g.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "big") % dim
        sign = 1.0 if (h[4] & 1) else -1.0
        vec[idx] += sign
    n2 = float(np.linalg.norm(vec))
    if n2 > 0:
        vec /= n2
    return vec


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class Embedder:
    """Embed text via an OpenAI-compatible backend, falling back to a deterministic local hashing embedder.

    Parameters mirror the LLM proxy: ``base_url`` + ``api_key`` + ``model``. Pass ``allow_remote=False`` (or leave
    the backend unreachable) to force the local fallback — the default in tests.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        allow_remote: bool = True,
        timeout: float = 60.0,
        local_dim: int = LOCAL_DIM,
    ):
        settings = get_settings()
        self.base_url = (base_url if base_url is not None else settings.llm_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.model = model or getattr(settings, "embedding_model", None) or "text-embedding-3-small"
        self.allow_remote = allow_remote
        self.timeout = timeout
        self.local_dim = local_dim
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        """Vector dimensionality (known after the first ``embed``; local fallback dim before that)."""
        return self._dim if self._dim is not None else self.local_dim

    def _embed_local(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.vstack([_hash_embed_one(t, self.local_dim) for t in texts]) if texts else np.zeros((0, self.local_dim))
        self._dim = self.local_dim
        return mat

    def _embed_remote(self, texts: Sequence[str]) -> np.ndarray | None:
        """Try the OpenAI-compatible backend. Returns ``None`` on any failure so the caller falls back locally."""
        import httpx  # lazy: keep import cost off the hot import path / optional in minimal installs

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.model, "input": list(texts)},
                    headers=headers,
                )
                r.raise_for_status()
                data = r.json()
        except Exception:
            return None
        rows = data.get("data")
        if not rows:
            return None
        try:
            ordered = sorted(rows, key=lambda d: d.get("index", 0))
            mat = np.asarray([d["embedding"] for d in ordered], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return None
        if mat.ndim != 2 or mat.shape[0] != len(texts):
            return None
        self._dim = mat.shape[1]
        return _l2_normalize(mat)

    def embed(self, texts: Sequence[str] | str) -> np.ndarray:
        """Embed one string or a sequence of strings → ``(n, dim)`` L2-normalised array."""
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim))
        if self.allow_remote:
            remote = self._embed_remote(texts)
            if remote is not None:
                return remote
        return self._embed_local(texts)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string → 1-D vector of length ``dim``."""
        return self.embed([text])[0]


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Process-wide embedder. Cached after first use; ``reset_embedder`` drops it for tests."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def reset_embedder() -> None:
    """Test hook: drop the cached embedder (e.g. after changing settings)."""
    global _embedder
    _embedder = None
