"""Rate limiting over a ``Cache`` backend, so the budget is enforced across gateway replicas when the cache
is Redis. Two algorithms:

  * fixed-window — one atomic counter per ``(key, window)``, armed with a TTL = window length; simple and
    cheap, the default.
  * token-bucket — a refilling bucket stored as ``(tokens, last_refill)``; smoother, allows controlled
    bursts up to ``capacity`` while refilling at ``rate`` tokens/sec.

Keyed by api-key / user id (the caller passes the identity). ``check`` returns a ``RateLimitResult`` with
``allowed`` + ``remaining`` + ``retry_after`` so the gateway can emit ``429`` + ``Retry-After``."""
from __future__ import annotations

import time
from dataclasses import dataclass

from .base import Cache, cache_key


@dataclass
class RateLimitResult:
    allowed: bool
    remaining: float
    limit: float
    retry_after: float = 0.0

    def headers(self) -> dict[str, str]:
        """OpenAI-style rate-limit headers for the gateway response."""
        h = {
            "X-RateLimit-Limit": str(int(self.limit)),
            "X-RateLimit-Remaining": str(max(0, int(self.remaining))),
        }
        if not self.allowed:
            h["Retry-After"] = str(int(self.retry_after) + 1)
        return h


class RateLimiter:
    """Fixed-window OR token-bucket limiter backed by a ``Cache``.

    Fixed window (default): ``limit`` requests per ``window`` seconds.
    Token bucket (``burst`` set): ``capacity = burst`` tokens, refilled at ``rate = limit / window`` per sec.
    """

    def __init__(
        self,
        cache: Cache,
        *,
        limit: int = 60,
        window: float = 60.0,
        burst: int | None = None,
        prefix: str = "rl:",
    ):
        self.cache = cache
        self.limit = limit
        self.window = window
        self.burst = burst
        self.prefix = prefix

    # --- fixed window ---
    def _window_key(self, identity: str) -> str:
        bucket = int(time.time() // self.window)
        return cache_key({"id": identity, "w": bucket}, prefix=self.prefix + "fw:")

    def _check_fixed_window(self, identity: str, cost: int) -> RateLimitResult:
        key = self._window_key(identity)
        count = self.cache.incr(key, cost, ttl=self.window)
        remaining = self.limit - count
        if count > self.limit:
            # seconds until this window rolls over
            retry = self.window - (time.time() % self.window)
            return RateLimitResult(False, max(0, remaining), self.limit, retry_after=retry)
        return RateLimitResult(True, max(0, remaining), self.limit)

    # --- token bucket ---
    def _bucket_key(self, identity: str) -> str:
        return cache_key({"id": identity}, prefix=self.prefix + "tb:")

    def _check_token_bucket(self, identity: str, cost: int) -> RateLimitResult:
        capacity = float(self.burst)
        rate = self.limit / self.window if self.window else float(self.limit)
        key = self._bucket_key(identity)
        now = time.time()
        state = self.cache.get(key) or {"tokens": capacity, "ts": now}
        elapsed = max(0.0, now - state.get("ts", now))
        tokens = min(capacity, state.get("tokens", capacity) + elapsed * rate)
        if tokens >= cost:
            tokens -= cost
            self.cache.set(key, {"tokens": tokens, "ts": now}, ttl=self.window * 4)
            return RateLimitResult(True, tokens, capacity)
        # not enough tokens: time to accrue the shortfall
        retry = (cost - tokens) / rate if rate > 0 else self.window
        self.cache.set(key, {"tokens": tokens, "ts": now}, ttl=self.window * 4)
        return RateLimitResult(False, tokens, capacity, retry_after=retry)

    def check(self, identity: str, *, cost: int = 1) -> RateLimitResult:
        """Consume ``cost`` units for ``identity``; return whether it is allowed + how much budget remains."""
        if not identity:
            identity = "anonymous"
        if self.burst is not None:
            return self._check_token_bucket(identity, cost)
        return self._check_fixed_window(identity, cost)
