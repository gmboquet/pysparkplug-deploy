"""HTTP routers grouped by concern: chat (OpenAI-compatible), models, accounts, mixle (distribution/decision),
feedback (RLHF), files (multimodal), mcp."""
from . import accounts, chat, feedback, files, mcp, mixle, models

__all__ = ["chat", "models", "accounts", "mixle", "feedback", "files", "mcp"]
