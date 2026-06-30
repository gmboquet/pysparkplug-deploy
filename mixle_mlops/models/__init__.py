"""Model adapters: the echo stub, the OpenAI-compatible LLM proxy, and (later) the mixle + composite adapters."""
from .echo import EchoAdapter
from .openai_compat import OpenAICompatAdapter

__all__ = ["EchoAdapter", "OpenAICompatAdapter"]
