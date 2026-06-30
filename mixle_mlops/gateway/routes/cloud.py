"""Cloud/object-store introspection routes. Reports which multi-cloud object-store backend is configured
(``MIXLE_OBJECT_STORE_URL``) and offers a tiny authenticated round-trip so an operator can verify cloud
credentials/wiring from EKS/AKS/GKE/ACK without shelling into a pod.

Wiring (integrator): ``app.include_router(cloud.router, prefix="/v1", tags=["cloud"])`` in ``gateway/app.py``.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ...accounts.models import User
from ...storage.objectstore import get_object_store
from ..auth import require_user

router = APIRouter()


class PutBody(BaseModel):
    key: str
    text: str


@router.get("/cloud/objectstore")
async def objectstore_status(user: User = Depends(require_user)):
    """Report the configured object-store backend (scheme, bucket/root, prefix) — no secrets."""
    store = get_object_store()
    return {
        "url": store.settings.url,
        "scheme": store.scheme,
        "protocol": store.protocol,
        "bucket": store.bucket,
        "prefix": store.prefix,
        "endpoint": store.settings.endpoint,
        "region": store.settings.region,
    }


@router.post("/cloud/objectstore/check")
async def objectstore_check(body: PutBody, user: User = Depends(require_user)):
    """Round-trip a small object (put → get → delete) to verify cloud credentials and connectivity."""
    store = get_object_store()
    key = f"_healthcheck/{user.id}/{body.key}"
    try:
        info = store.put(key, body.text.encode("utf-8"))
        got = store.get(key).decode("utf-8")
        url = store.url(key)
        store.delete(key)
    except Exception as exc:  # surface driver/cred errors as a 502 rather than a 500 stack
        raise HTTPException(status_code=502, detail=f"object-store check failed: {exc}")
    return {"ok": got == body.text, "uri": info.uri, "url": url, "size": info.size}
