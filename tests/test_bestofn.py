"""Best-of-N self-consistency: answer extraction, majority voting + calibrated confidence, and the
X-Self-Consistency header through the chat route."""
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
    ChoiceDelta,
    ModelAdapter,
)
from mixle_mlops.gateway.bestofn import best_of_n, extract_answer


class SeqAdapter(ModelAdapter):
    """Returns a fixed cycle of texts — deterministic stand-in for a sampler at temperature>0."""
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


def test_extract_answer():
    assert extract_answer("the result is \\boxed{42} ok") == "42"
    assert extract_answer("blah blah\nFinal answer: 17.") == "17"
    assert extract_answer("first 3 then 99") == "99"
    assert extract_answer("Hello   World") == "hello world"


def test_best_of_n_majority_and_confidence():
    from mixle_mlops.core.adapters import ChatRequest

    adapter = SeqAdapter("voter", ["answer: 42", "answer: 42", "answer: 7"])
    req = ChatRequest(model="voter", messages=[ChatMessage(role="user", content="2*21?")])
    completion, info = asyncio.run(best_of_n(adapter, req, n=3))
    assert info["answer"] == "42" and info["votes"] == 2
    assert abs(info["confidence"] - 2 / 3) < 1e-9 and info["distinct"] == 2
    assert "42" in completion.choices[0].message.text()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    from mixle_mlops.config import get_settings
    from mixle_mlops.gateway.app import create_app

    get_settings.cache_clear()
    db._engine = None
    app = create_app()
    with TestClient(app) as c:
        app.state.registry.register(SeqAdapter("voter", ["answer: 42", "answer: 42", "answer: 7"]))
        yield c
    get_settings.cache_clear()
    db._engine = None


def test_best_of_n_over_http_sets_confidence_header(client):
    raw = client.post("/auth/signup", json={"email": "b@t.com", "password": "pw12345"}).json()["api_key"]
    headers = {"Authorization": f"Bearer {raw}"}
    r = client.post("/v1/chat/completions", headers=headers,
                    json={"model": "voter", "extra": {"best_of_n": 3},
                          "messages": [{"role": "user", "content": "2*21?"}]})
    assert r.status_code == 200
    assert "42" in r.json()["choices"][0]["message"]["content"]
    assert abs(float(r.headers["X-Self-Consistency"]) - 2 / 3) < 0.01
