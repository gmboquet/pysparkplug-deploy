"""Core model contract + registry (backend-agnostic)."""
from .adapters import (
    CapabilityError,
    ChatChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    ChatRequest,
    ChoiceDelta,
    ChatChunkChoice,
    ModelAdapter,
    ModelInfo,
    Usage,
)
from .registry import ModelRegistry

__all__ = [
    "ModelAdapter", "ModelInfo", "ModelRegistry", "ChatMessage", "ChatRequest", "ChatCompletion",
    "ChatChoice", "ChatCompletionChunk", "ChatChunkChoice", "ChoiceDelta", "Usage", "CapabilityError",
]
