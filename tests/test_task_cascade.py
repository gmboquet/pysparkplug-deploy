"""Serve a distilled task cascade through the gateway: predict / score / decide / chat with an honest escalate.

Self-contained: builds the app via create_app(), includes the mixle router, and registers the demo distilled
model on the live registry. Skipped when torch is unavailable (the student needs it).
"""

import mixle_mlops.storage.db as db
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("torch")
pytest.importorskip("safetensors")

from mixle_mlops.config import get_settings  # noqa: E402
from mixle_mlops.gateway.app import create_app  # noqa: E402
from mixle_mlops.gateway.routes import mixle as mixle_routes  # noqa: E402
from mixle_mlops.models.task_cascade import TaskCascadeAdapter, register_demo_task_model  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    app = create_app()
    app.include_router(mixle_routes.router, prefix="/v1")
    with TestClient(app) as c:
        register_demo_task_model(c.app.state.registry, name="demo-task")
        yield c
    get_settings.cache_clear()
    db._engine = None


def _key(client, email):
    return client.post("/auth/signup", json={"email": email, "password": "pw12345"}).json()["api_key"]


def _h(client, email="u@example.com"):
    return {"Authorization": f"Bearer {_key(client, email)}"}


def test_capabilities(client):
    r = client.get("/v1/mixle/capabilities/demo-task", headers=_h(client))
    assert r.status_code == 200
    caps = set(r.json()["capabilities"])
    assert {"predict", "score", "chat"} <= caps
    assert "decide" not in caps  # the mixle /decide route is the loss/action surface, not a classifier task


def test_predict_returns_labels_and_decisions(client):
    body = {"model": "demo-task", "records": ["free prize winner click", "team meeting report today"]}
    r = client.post("/v1/mixle/predict", json=body, headers=_h(client))
    assert r.status_code == 200
    out = r.json()
    assert out["labels"] == ["spam", "ham"]
    # calibrated -> each record carries an honest answer-or-escalate decision
    assert "escalation_rate" in out
    for d in out["decisions"]:
        assert "escalate" in d
        assert (d["label"] is None) == d["escalate"]


def test_score_returns_proba_over_labels(client):
    body = {"model": "demo-task", "records": ["free prize"]}
    r = client.post("/v1/mixle/score", json=body, headers=_h(client))
    assert r.status_code == 200
    out = r.json()
    assert set(out["labels"]) == {"spam", "ham"}
    assert abs(sum(out["proba"][0]) - 1.0) < 1e-5


def test_chat_classifies_last_message(client):
    body = {"model": "demo-task", "messages": [{"role": "user", "content": "free prize winner click now"}]}
    r = client.post("/v1/chat/completions", json=body, headers=_h(client))
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert "demo-task" in content


def test_cascade_routes_on_calibrated_signal():
    """The distilled task model's conformal/density gate drives the FrugalGPT cascade (not best-of-N voting)."""
    import asyncio

    from mixle_mlops.core.adapters import (
        ChatChoice,
        ChatCompletion,
        ChatMessage,
        ChatRequest,
        ModelAdapter,
    )
    from mixle_mlops.gateway.cascade import cascade
    from mixle_mlops.models.task_cascade import register_demo_task_model

    class _Reg:
        def __init__(self):
            self.a = None

        def register(self, adapter):
            self.a = adapter

    local = register_demo_task_model(_Reg(), name="local-task")

    class FrontierStub(ModelAdapter):  # a frontier LLM that always answers "FRONTIER"
        kind = "llm"

        @property
        def name(self):
            return "frontier"

        async def stream(self, req):  # pragma: no cover - chat() is overridden, so stream() is unused
            return
            yield  # make this an async generator

        async def chat(self, req):
            msg = ChatMessage(role="assistant", content="FRONTIER")
            return ChatCompletion(model=req.model, choices=[ChatChoice(message=msg)])

    frontier = FrontierStub()

    def run(text):
        req = ChatRequest(model="local-task", messages=[ChatMessage(role="user", content=text)])
        return asyncio.run(cascade(local, frontier, req))

    # a clearly-spam input is answered locally (no escalation)
    comp, info = run("free prize winner click now")
    assert info["signal"] == "calibrated"
    assert info["escalated"] is False
    assert comp.choices[0].message.content == "spam"

    # a gibberish / out-of-distribution input escalates to the frontier
    comp2, info2 = run("zxcvb 12345 ΩΨΔ !!! qqqq")
    assert info2["escalated"] is True
    assert comp2.choices[0].message.content == "FRONTIER"


def test_serve_extraction_model():
    """An extraction task model (text -> {field: value}) is servable: predict returns dicts, no score capability."""
    import asyncio

    from mixle.task.extract import distill_extractor
    from mixle.task.model import TaskModel  # noqa: F811

    fields = ["id", "amount"]

    def teacher(texts):
        out = []
        for t in texts:
            d = {}
            for tok in t.split():
                if tok.startswith("INV-"):
                    d["id"] = tok[4:]
                if tok.startswith("$"):
                    d["amount"] = tok[1:]
            out.append(d)
        return out

    lines = [f"INV-{1000 + i} paid ${i}.50 today" for i in range(120)]
    extractor: TaskModel = distill_extractor(teacher, lines, fields, epochs=120, seed=0)
    adapter = TaskCascadeAdapter("extract", extractor)

    caps = adapter.capabilities()
    assert "predict" in caps and "score" not in caps  # extractors have no class probabilities
    out = asyncio.run(adapter.predict(["INV-1055 paid $7.50 today"]))
    assert out["results"][0].get("id") == "1055"
    assert "labels" not in out  # results are dicts, not labels


# --- unit level: the adapter directly, no gateway ---
def test_adapter_plain_taskmodel_capabilities():
    import numpy as np

    from mixle.task.distill import distill

    def teacher(texts):
        return ["spam" if "free" in t else "ham" for t in texts]

    rng = np.random.RandomState(0)
    texts = [("free " if rng.rand() < 0.5 else "") + "hello world today" for _ in range(80)]
    student = distill(teacher, texts, n=3, dim=128, hidden=[16], epochs=80, seed=0)
    adapter = TaskCascadeAdapter("plain", student)  # a plain TaskModel, not calibrated
    assert {"predict", "score", "chat"} <= adapter.capabilities()
    import asyncio

    out = asyncio.run(adapter.predict(["free stuff"]))
    assert out["labels"] == ["spam"]
    assert "decisions" not in out  # a plain (uncalibrated) model carries no escalate decision


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
