"""The catalog of hosted models — mixle artifacts + configured LLM backends — keyed by id, listed to
``/v1/models``. A process-wide registry built at startup from config and (later) the persistent model store."""
from __future__ import annotations

from .adapters import ModelAdapter, ModelInfo


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ModelAdapter] = {}

    def register(self, adapter: ModelAdapter) -> ModelAdapter:
        self._models[adapter.name] = adapter
        return adapter

    def get(self, name: str) -> ModelAdapter:
        try:
            return self._models[name]
        except KeyError:
            raise KeyError(f"model {name!r} not found; available: {self.names()}")

    def has(self, name: str) -> bool:
        return name in self._models

    def list(self) -> list[ModelInfo]:
        return [a.info() for a in self._models.values()]

    def names(self) -> list[str]:
        return sorted(self._models)
