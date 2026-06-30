"""Per-model backends: a local + a cloud frontier model hosted in one registry (the cascade prerequisite)."""
import json

from mixle_mlops.config import Settings, get_settings
from mixle_mlops.gateway.app import build_registry


def test_llm_backends_registered(monkeypatch):
    monkeypatch.setenv("MIXLE_LLM_BACKENDS", json.dumps({
        "local": {"base_url": "http://ollama:11434/v1"},
        "frontier": {"base_url": "https://api.example/v1", "api_key": "k", "upstream_model": "gpt-4o"},
    }))
    get_settings.cache_clear()
    try:
        registry = build_registry(Settings())
        assert registry.has("local") and registry.has("frontier")
        frontier = registry.get("frontier")
        assert frontier.base_url == "https://api.example/v1" and frontier.upstream_model == "gpt-4o"
    finally:
        get_settings.cache_clear()
