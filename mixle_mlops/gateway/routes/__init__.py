"""HTTP routers grouped by concern: chat (OpenAI-compatible), models, accounts (+ later: mixle, feedback, mcp)."""
from . import accounts, chat, models

__all__ = ["chat", "models", "accounts"]
