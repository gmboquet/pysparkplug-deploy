"""OpenAI-compatible chat completions — the product's main inference route. Composes the platform pipeline:
rate-limit → multimodal-normalize → RAG-augment → response-cache → dispatch → persist (conversation memory).
Each stage is gated/opt-in/defensive so the default path stays simple and the extras never break a chat."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...accounts.models import User
from ...config import get_settings
from ...core.adapters import ChatCompletion, ChatRequest
from ...multimodal.content import MultimodalError, normalize_messages
from ..auth import current_user

router = APIRouter()


def _principal(user: User | None, request: Request) -> str:
    if user is not None:
        return user.id
    return request.client.host if request.client else "anon"


def _persist(user: User | None, req: ChatRequest, name: str, assistant_text: str) -> None:
    """Record the turn into the user's conversation history (memory + export). Best-effort."""
    if user is None:
        return
    try:
        from sqlmodel import Session

        from ...conversations.service import persist_turn
        from ...storage.db import get_engine

        user_text = req.messages[-1].text() if req.messages else ""
        with Session(get_engine()) as session:
            persist_turn(session, user.id, req.extra.get("conversation_id"), user_text, assistant_text, model=name)
    except Exception:
        pass


@router.post("/chat/completions")
async def chat_completions(req: ChatRequest, request: Request, user: User | None = Depends(current_user)):
    settings = get_settings()
    registry = request.app.state.registry
    name = req.model or settings.default_model
    if not registry.has(name):
        raise HTTPException(status_code=404, detail=f"model {name!r} not found")
    adapter = registry.get(name)

    # 1. rate limit (opt-in via MIXLE_RATE_LIMIT_PER_MIN), shared across replicas when Redis is configured
    if settings.rate_limit_per_min > 0:
        try:
            from ...cache import RateLimiter, get_cache

            res = RateLimiter(get_cache(), limit=settings.rate_limit_per_min, window=60).check(_principal(user, request))
            if not res.allowed:
                raise HTTPException(status_code=429, detail="rate limit exceeded", headers=res.headers())
        except HTTPException:
            raise
        except Exception:
            pass

    # 2. resolve uploaded-file refs → image_url parts for vision LLMs
    try:
        req.messages = normalize_messages(req.messages)
    except MultimodalError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # 3. RAG: augment with retrieved context from this user's past conversations + documents (opt-in via extra.rag)
    if user is not None and req.extra.get("rag"):
        try:
            from ...rag.augment import build_rag_messages

            req.messages = build_rag_messages(user.id, req.messages)
        except Exception:
            pass

    # 4. response cache (opt-in via MIXLE_ENABLE_RESPONSE_CACHE), exact-match, non-streaming only
    rc = None
    if settings.enable_response_cache and not req.stream:
        try:
            from ...cache import ResponseCache, get_cache

            rc = ResponseCache(get_cache())
            hit = rc.get(req)
            if hit is not None:
                return ChatCompletion.model_validate(hit)
        except Exception:
            rc = None

    if req.stream:
        async def event_stream():
            buf: list[str] = []
            try:
                async for chunk in adapter.stream(req):
                    for ch in chunk.choices:
                        if ch.delta.content:
                            buf.append(ch.delta.content)
                    yield f"data: {chunk.model_dump_json()}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n"
            _persist(user, req, name, "".join(buf))
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    completion = await adapter.chat(req)
    if rc is not None:
        try:
            rc.set(req, completion.model_dump())
        except Exception:
            pass
    _persist(user, req, name, completion.choices[0].message.text() if completion.choices else "")
    return completion
