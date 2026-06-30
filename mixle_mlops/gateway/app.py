"""The platform gateway: a FastAPI app exposing the OpenAI-compatible API + the platform API, with a model
registry built from config. Runnable end-to-end against the echo model or any OpenAI-compatible LLM backend."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import Settings, get_settings
from ..core.registry import ModelRegistry
from ..models import EchoAdapter, OpenAICompatAdapter
from ..storage.db import init_db
from .routes import accounts, chat, models


def build_registry(settings: Settings) -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(EchoAdapter("echo"))                      # always available; zero backends needed
    for model_id in settings.llm_models:                        # configured LLM backend models
        registry.register(OpenAICompatAdapter(model_id, base_url=settings.llm_base_url,
                                              api_key=settings.llm_api_key))
    return registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_db()
    app.state.settings = settings
    app.state.registry = build_registry(settings)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="mixle-mlops", version="0.1.0",
                  description="All-in-one AI platform: host mixle + open LLMs, OpenAI-compatible.",
                  lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"],
        allow_headers=["*"], allow_credentials=True,
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "mixle-mlops"}

    app.include_router(accounts.router, tags=["accounts"])
    app.include_router(models.router, prefix="/v1", tags=["models"])
    app.include_router(chat.router, prefix="/v1", tags=["chat"])
    return app


app = create_app()
