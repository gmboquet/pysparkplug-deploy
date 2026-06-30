"""Integration: the verifier best-of-N, MoA focal-diversity (select_k), and constrained-decoding chat hooks."""
import mixle_mlops.storage.db as db
import pytest
from fastapi.testclient import TestClient

from mixle_mlops.core.adapters import (
    ChatChoice,
    ChatChunkChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    ChoiceDelta,
    ModelAdapter,
)
from mixle_mlops.config import get_settings
from mixle_mlops.gateway.app import create_app


class SeqAdapter(ModelAdapter):
    kind = "llm"

    def __init__(self, name, texts):
        self._name, self._texts, self._i = name, list(texts), 0

    @property
    def name(self):
        return self._name

    async def chat(self, req):
        t = self._texts[self._i % len(self._texts)]
        self._i += 1
        return ChatCompletion(model=req.model, choices=[ChatChoice(
            message=ChatMessage(role="assistant", content=t), finish_reason="stop")])

    async def stream(self, req):
        c = await self.chat(req)
        yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(
            delta=ChoiceDelta(content=c.choices[0].message.text()), finish_reason="stop")])


class ReflectAdapter(ModelAdapter):
    kind = "llm"

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    async def chat(self, req):
        last = req.messages[-1].text() if req.messages else ""
        return ChatCompletion(model=req.model, choices=[ChatChoice(
            message=ChatMessage(role="assistant", content=f"AGG::{last}"), finish_reason="stop")])

    async def stream(self, req):
        c = await self.chat(req)
        yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(
            delta=ChoiceDelta(content=c.choices[0].message.text()), finish_reason="stop")])


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    app = create_app()
    with TestClient(app) as c:
        app.state.registry.register(SeqAdapter("gen", ["answer: 7", "answer: 42", "answer: 13"]))
        for n in ("p1", "p2", "p3"):
            app.state.registry.register(SeqAdapter(n, [f"from {n}"]))
        app.state.registry.register(ReflectAdapter("agg"))
        app.state.registry.register(SeqAdapter("jsongen", ['{"name": "x", "age": 5}']))
        yield c
    get_settings.cache_clear()
    db._engine = None


def _key(c, email):
    raw = c.post("/auth/signup", json={"email": email, "password": "pw12345"}).json()["api_key"]
    return {"Authorization": f"Bearer {raw}"}


def test_best_of_n_with_numeric_verifier(client):
    h = _key(client, "v@t.com")
    r = client.post("/v1/chat/completions", headers=h, json={
        "model": "gen", "messages": [{"role": "user", "content": "6*7?"}],
        "extra": {"best_of_n": 3, "verifier": {"type": "numeric", "spec": {"op": "eval", "expr": "6*7"}}}})
    assert r.status_code == 200
    assert "42" in r.json()["choices"][0]["message"]["content"]
    assert float(r.headers["X-Verifier-Score"]) == 1.0


def test_moa_with_focal_diversity_select_k(client):
    h = _key(client, "moa@t.com")
    r = client.post("/v1/chat/completions", headers=h, json={
        "model": "agg", "messages": [{"role": "user", "content": "q"}],
        "extra": {"moa": {"proposers": ["p1", "p2", "p3"], "aggregator": "agg", "select_k": 2}}})
    assert r.status_code == 200 and "AGG::" in r.json()["choices"][0]["message"]["content"]


def test_constrained_validates_json(client):
    h = _key(client, "c@t.com")
    r = client.post("/v1/chat/completions", headers=h, json={
        "model": "jsongen", "messages": [{"role": "user", "content": "give me a person"}],
        "extra": {"constrained": {"json_schema": {"type": "object", "required": ["name", "age"],
                                                  "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}}}})
    assert r.status_code == 200 and r.headers["X-Constrained-Valid"] == "1"
