"""Cascade router: escalate to the frontier model when the local model's self-consistency is low, keep the
local answer when it is high, and the routing headers through the chat route."""
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
from mixle_mlops.gateway.cascade import cascade


class SeqAdapter(ModelAdapter):
    kind = "llm"

    def __init__(self, name, texts):
        self._name = name
        self._texts = list(texts)
        self._i = 0

    @property
    def name(self):
        return self._name

    async def chat(self, req):
        text = self._texts[self._i % len(self._texts)]
        self._i += 1
        return ChatCompletion(model=req.model,
                              choices=[ChatChoice(message=ChatMessage(role="assistant", content=text),
                                                  finish_reason="stop")])

    async def stream(self, req):
        completion = await self.chat(req)
        yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(
            delta=ChoiceDelta(content=completion.choices[0].message.text()), finish_reason="stop")])


_REQ = ChatRequest(model="local", messages=[ChatMessage(role="user", content="q")])


def test_cascade_escalates_on_low_confidence():
    local = SeqAdapter("local", ["answer: 1", "answer: 2", "answer: 3", "answer: 4"])   # all distinct -> conf 0.25
    frontier = SeqAdapter("frontier", ["answer: 42"])
    completion, info = asyncio.run(cascade(local, frontier, _REQ, threshold=0.6, n=4))
    assert info["escalated"] and info["frontier_model"] == "frontier"
    assert "42" in completion.choices[0].message.text()


def test_cascade_keeps_local_on_high_confidence():
    local = SeqAdapter("local", ["answer: 7"])                  # always agrees -> conf 1.0
    frontier = SeqAdapter("frontier", ["answer: 42"])
    completion, info = asyncio.run(cascade(local, frontier, _REQ, threshold=0.6, n=4))
    assert not info["escalated"] and info["frontier_model"] is None
    assert "7" in completion.choices[0].message.text()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    from mixle_mlops.config import get_settings
    from mixle_mlops.gateway.app import create_app

    get_settings.cache_clear()
    db._engine = None
    app = create_app()
    with TestClient(app) as c:
        app.state.registry.register(SeqAdapter("local", ["answer: 1", "answer: 2", "answer: 3", "answer: 4"]))
        app.state.registry.register(SeqAdapter("frontier", ["answer: 42"]))
        yield c
    get_settings.cache_clear()
    db._engine = None


def test_cascade_over_http_escalates(client):
    raw = client.post("/auth/signup", json={"email": "c@t.com", "password": "pw12345"}).json()["api_key"]
    headers = {"Authorization": f"Bearer {raw}"}
    r = client.post("/v1/chat/completions", headers=headers,
                    json={"model": "local", "extra": {"cascade": {"frontier": "frontier", "threshold": 0.6, "n": 4}},
                          "messages": [{"role": "user", "content": "hard q"}]})
    assert r.status_code == 200
    assert r.headers["X-Cascade-Escalated"] == "1"
    assert "42" in r.json()["choices"][0]["message"]["content"]
