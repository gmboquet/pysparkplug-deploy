"""OpenAI-compatible chat completions (streaming SSE + non-streaming). The product's main inference route —
any OpenAI client/SDK/UI can talk to it, and mixle / LLM / composite models are all reachable here."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...accounts.models import User
from ...config import get_settings
from ...core.adapters import ChatRequest
from ...multimodal.content import MultimodalError, normalize_messages
from ..auth import current_user

router = APIRouter()


@router.post("/chat/completions")
async def chat_completions(req: ChatRequest, request: Request, user: User | None = Depends(current_user)):
    registry = request.app.state.registry
    name = req.model or get_settings().default_model
    if not registry.has(name):
        raise HTTPException(status_code=404, detail=f"model {name!r} not found")
    adapter = registry.get(name)
    try:                                              # resolve uploaded-file refs → image_url parts for vision LLMs
        req.messages = normalize_messages(req.messages)
    except MultimodalError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if req.stream:
        async def event_stream():
            try:
                async for chunk in adapter.stream(req):
                    yield f"data: {chunk.model_dump_json()}\n\n"
            except Exception as exc:  # surface backend errors to the client in-band
                yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    completion = await adapter.chat(req)
    return completion
