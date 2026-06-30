"""The HTTP gateway (FastAPI): OpenAI-compatible + platform API over the model registry."""
from .app import app, create_app

__all__ = ["create_app", "app"]
