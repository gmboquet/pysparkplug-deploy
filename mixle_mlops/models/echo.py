"""A dependency-free echo model, so the platform runs end-to-end with no LLM backend (tests + local dev)."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from ..core.adapters import ChatChunkChoice, ChatCompletionChunk, ChatRequest, ChoiceDelta, ModelAdapter


class EchoAdapter(ModelAdapter):
    kind = "llm"

    def __init__(self, name: str = "echo"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        last = next((m for m in reversed(req.messages) if m.role == "user"), None)
        reply = f"echo: {last.text() if last else ''}".strip()
        cid = "chatcmpl-echo"
        yield ChatCompletionChunk(id=cid, model=req.model,
                                  choices=[ChatChunkChoice(delta=ChoiceDelta(role="assistant"))])
        for tok in reply.split(" "):
            await asyncio.sleep(0)
            yield ChatCompletionChunk(id=cid, model=req.model,
                                      choices=[ChatChunkChoice(delta=ChoiceDelta(content=tok + " "))])
        yield ChatCompletionChunk(id=cid, model=req.model,
                                  choices=[ChatChunkChoice(delta=ChoiceDelta(), finish_reason="stop")])
