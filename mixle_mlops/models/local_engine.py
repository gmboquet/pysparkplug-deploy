"""Serve a local model through the logit-level decode engine — so token-level Product-of-Experts and grammar
masking are available as a *hosted model*, not just a library call.

One provider → a single local model. Several providers → a Product-of-Experts ensemble decoded token-by-token
(true per-step fusion, the thing the chat-API proxy cannot do). This is how the platform hosts a small local
model with the bridge's logit-level levers turned on."""
from __future__ import annotations

from typing import Any, AsyncIterator

from ..core.adapters import (
    ChatChoice,
    ChatChunkChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    ChatRequest,
    ChoiceDelta,
    ModelAdapter,
)
from ..engines import decode


class LocalEngineAdapter(ModelAdapter):
    kind = "llm"

    def __init__(self, name: str, providers: Any, *, max_new_tokens: int = 128,
                 weights: list[float] | None = None):
        self._name = name
        self._providers = list(providers) if isinstance(providers, (list, tuple)) else [providers]
        self._primary = self._providers[0]
        self.max_new_tokens = max_new_tokens
        self.weights = weights

    @property
    def name(self) -> str:
        return self._name

    def vocab(self) -> dict[int, str]:
        return self._primary.vocab()

    def _prompt_ids(self, req: ChatRequest) -> list[int]:
        text = "".join(f"{m.role}: {m.text()}\n" for m in req.messages) + "assistant: "
        return self._primary.encode(text)

    def _generate(self, req: ChatRequest) -> str:
        ids = self._prompt_ids(req)
        provs = self._providers if len(self._providers) > 1 else self._primary
        out = decode(
            provs, prompt_ids=ids, max_new_tokens=req.max_tokens or self.max_new_tokens,
            weights=self.weights, eos_id=getattr(self._primary, "eos", None),
            greedy=not req.temperature, temperature=req.temperature or 1.0, top_p=req.top_p or 1.0,
            grammar=req.extra.get("_grammar"),                # an optional pre-built TokenFSA (programmatic)
        )
        return self._primary.decode_text(out)

    async def chat(self, req: ChatRequest) -> ChatCompletion:
        text = self._generate(req)
        return ChatCompletion(model=req.model, choices=[ChatChoice(
            message=ChatMessage(role="assistant", content=text), finish_reason="stop")])

    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        text = self._generate(req)
        yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(
            delta=ChoiceDelta(role="assistant", content=text), finish_reason="stop")])


def load_local_engine(name: str, model_names: list[str], *, max_new_tokens: int = 128) -> LocalEngineAdapter:
    """Load one or more transformers models and wrap them as a local-engine adapter (PoE ensemble if >1)."""
    from ..engines import HFLogitProvider

    providers = [HFLogitProvider(model_name=m) for m in model_names]
    return LocalEngineAdapter(name, providers, max_new_tokens=max_new_tokens)
