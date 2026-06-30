"""Mixture-of-Agents: proposers' answers reach the aggregator, which synthesizes the final response."""
import asyncio

import mixle_mlops.storage.db as db
import pytest
from fastapi.testclient import TestClient

from mixle_mlops.core.adapters import (
    ChatChoice,
    ChatChunkChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    ChatRequest,
    ChoiceDelta,
    ModelAdapter,
)
from mixle_mlops.gateway.moa import mixture_of_agents


class FixedAdapter(ModelAdapter):
    kind = "llm"

    def __init__(self, name, text):
        self._name = name
        self._fixed = text

    @property
    def name(self):
        return self._name

    async def chat(self, req):
        return ChatCompletion(model=req.model, choices=[ChatChoice(
            message=ChatMessage(role="assistant", content=self._fixed), finish_reason="stop")])

    async def stream(self, req):
        yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(
            delta=ChoiceDelta(content=self._fixed), finish_reason="stop")])


class ReflectAdapter(ModelAdapter):
    """Returns the last user message it received — proves the proposals reached the aggregator."""
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
        completion = await self.chat(req)
        yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(
            delta=ChoiceDelta(content=completion.choices[0].message.text()), finish_reason="stop")])


_REQ = ChatRequest(model="agg", messages=[ChatMessage(role="user", content="name an animal")])


def test_moa_aggregator_sees_all_proposals():
    proposers = [FixedAdapter("p1", "cats"), FixedAdapter("p2", "dogs"), FixedAdapter("p3", "birds")]
    completion, info = asyncio.run(mixture_of_agents(proposers, ReflectAdapter("agg"), _REQ, layers=1))
    text = completion.choices[0].message.text()
    assert all(animal in text for animal in ("cats", "dogs", "birds"))
    assert info["n_proposals"] == 3 and info["aggregator"] == "agg" and info["layers"] == 1


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    from mixle_mlops.config import get_settings
    from mixle_mlops.gateway.app import create_app

    get_settings.cache_clear()
    db._engine = None
    app = create_app()
    with TestClient(app) as c:
        for n, t in [("p1", "cats"), ("p2", "dogs"), ("p3", "birds")]:
            app.state.registry.register(FixedAdapter(n, t))
        app.state.registry.register(ReflectAdapter("agg"))
        yield c
    get_settings.cache_clear()
    db._engine = None


def test_moa_over_http(client):
    raw = client.post("/auth/signup", json={"email": "m@t.com", "password": "pw12345"}).json()["api_key"]
    headers = {"Authorization": f"Bearer {raw}"}
    r = client.post("/v1/chat/completions", headers=headers,
                    json={"model": "agg",
                          "extra": {"moa": {"proposers": ["p1", "p2", "p3"], "aggregator": "agg"}},
                          "messages": [{"role": "user", "content": "name an animal"}]})
    assert r.status_code == 200
    text = r.json()["choices"][0]["message"]["content"]
    assert all(animal in text for animal in ("cats", "dogs", "birds"))
