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


# --- unit level: the adapter directly, no gateway ---
def test_adapter_plain_taskmodel_has_no_decide():
    import numpy as np

    from mixle.task.distill import distill

    def teacher(texts):
        return ["spam" if "free" in t else "ham" for t in texts]

    rng = np.random.RandomState(0)
    texts = [("free " if rng.rand() < 0.5 else "") + "hello world today" for _ in range(80)]
    student = distill(teacher, texts, n=3, dim=128, hidden=[16], epochs=80, seed=0)
    adapter = TaskCascadeAdapter("plain", student)  # a plain TaskModel, not calibrated
    assert "decide" not in adapter.capabilities()
    assert "predict" in adapter.capabilities()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
