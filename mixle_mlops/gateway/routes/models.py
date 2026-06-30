"""Model catalog (OpenAI-compatible /v1/models) — lists hosted mixle + LLM + composite models and their
capabilities, so a client can discover which support the mixle distribution/decision routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ...accounts.models import User
from ..auth import current_user

router = APIRouter()


@router.get("/models")
async def list_models(request: Request, user: User | None = Depends(current_user)):
    return {"object": "list", "data": [m.model_dump() for m in request.app.state.registry.list()]}


@router.get("/models/{model_id}")
async def get_model(model_id: str, request: Request, user: User | None = Depends(current_user)):
    registry = request.app.state.registry
    if not registry.has(model_id):
        raise HTTPException(status_code=404, detail=f"model {model_id!r} not found")
    return registry.get(model_id).info().model_dump()
