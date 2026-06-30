"""The platform gateway: a FastAPI app exposing the OpenAI-compatible API + the platform API, with a model
registry built from config. Runnable end-to-end against the echo model or any OpenAI-compatible LLM backend."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import Settings, get_settings
from ..core.registry import ModelRegistry
from ..image_gen import register_demo_image_model
from ..models import EchoAdapter, OpenAICompatAdapter
from ..models.mixle_model import register_demo_mixle_model
from ..storage.db import init_db
from .routes import (
    accounts,
    cache,
    chat,
    cloud,
    conversations,
    datasets,
    feedback,
    files,
    images,
    mcp,
    mixle,
    models,
    rag,
)


def build_registry(settings: Settings) -> ModelRegistry:
    registry = ModelRegistry()
    registry.register(EchoAdapter("echo"))                      # always available; zero backends needed
    for model_id in settings.llm_models:                        # configured LLM backend models
        registry.register(OpenAICompatAdapter(model_id, base_url=settings.llm_base_url,
                                              api_key=settings.llm_api_key))
    if settings.enable_demo_models:                             # demo models for /v1/mixle and /v1/images
        for _register in (register_demo_mixle_model, register_demo_image_model):
            try:
                _register(registry)
            except Exception:                                   # never let a demo registration break startup
                pass
    if settings.image_model and settings.image_base_url:        # a real image backend, if configured
        try:
            from ..image_gen import ImageGenAdapter
            registry.register(ImageGenAdapter(settings.image_model, backend="openai",
                                              base_url=settings.image_base_url, api_key=settings.image_api_key))
        except Exception:
            pass
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
    app.include_router(mixle.router, prefix="/v1", tags=["mixle"])        # /v1/mixle/predict|score|decide|latent|capabilities
    app.include_router(feedback.router, prefix="/v1", tags=["feedback"])  # /v1/feedback, /v1/rlhf/*
    app.include_router(files.router, prefix="/v1", tags=["files"])        # /v1/files (multimodal uploads)
    app.include_router(mcp.router, tags=["mcp"])                          # /mcp (JSON-RPC over HTTP)
    app.include_router(rag.router, prefix="/v1", tags=["rag"])            # /v1/documents, /v1/rag/search
    app.include_router(cache.router, prefix="/v1", tags=["cache"])        # /v1/cache/stats
    app.include_router(conversations.router, prefix="/v1", tags=["conversations"])  # /v1/conversations (+ export)
    app.include_router(images.router, prefix="/v1", tags=["images"])      # /v1/images/generations
    app.include_router(datasets.router, prefix="/v1", tags=["datasets"])  # /v1/datasets/generate
    app.include_router(cloud.router, prefix="/v1", tags=["cloud"])        # /v1/cloud/objectstore
    return app


app = create_app()
