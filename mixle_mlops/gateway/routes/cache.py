"""Cache + rate-limit introspection route.

  * ``GET /v1/cache/stats`` — backend kind + best-effort stats (size/hits/misses for memory; memory/keys for
    Redis). Authenticated (``require_user``).

The cache itself is wired into the chat pipeline by the integrator (see this package's report); this router
only exposes operational visibility."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from ...accounts.models import User
from ...cache import get_cache
from ..auth import require_user

router = APIRouter()


@router.get("/cache/stats")
def cache_stats(user: User = Depends(require_user)) -> dict[str, Any]:
    cache = get_cache()
    return {"object": "cache.stats", "stats": cache.stats()}
