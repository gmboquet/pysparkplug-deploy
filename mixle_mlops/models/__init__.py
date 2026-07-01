"""Model adapters: the echo stub, the OpenAI-compatible LLM proxy, the mixle model, and the distilled task cascade."""
from .echo import EchoAdapter
from .openai_compat import OpenAICompatAdapter
from .task_cascade import TaskCascadeAdapter, register_demo_task_model

__all__ = ["EchoAdapter", "OpenAICompatAdapter", "TaskCascadeAdapter", "register_demo_task_model"]
