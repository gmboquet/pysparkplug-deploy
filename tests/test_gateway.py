"""End-to-end test of the platform foundation: signup → API key → OpenAI-compatible chat (echo model)."""
import mixle_mlops.storage.db as db
import pytest
from fastapi.testclient import TestClient

from mixle_mlops.config import get_settings
from mixle_mlops.gateway.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None                       # fresh sqlite under tmp_path
    app = create_app()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
    db._engine = None


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_auth_and_chat(client):
    raw = client.post("/auth/signup", json={"email": "t@t.com", "password": "pw12345"}).json()["api_key"]
    headers = {"Authorization": f"Bearer {raw}"}
    assert client.get("/v1/models").status_code == 401                      # auth required
    assert "echo" in [m["id"] for m in client.get("/v1/models", headers=headers).json()["data"]]
    r = client.post("/v1/chat/completions", headers=headers,
                    json={"model": "echo", "messages": [{"role": "user", "content": "hi there"}]})
    assert r.status_code == 200
    assert "echo: hi there" in r.json()["choices"][0]["message"]["content"]


def test_streaming(client):
    raw = client.post("/auth/signup", json={"email": "s@t.com", "password": "pw12345"}).json()["api_key"]
    headers = {"Authorization": f"Bearer {raw}"}
    with client.stream("POST", "/v1/chat/completions", headers=headers,
                       json={"model": "echo", "stream": True,
                             "messages": [{"role": "user", "content": "go"}]}) as s:
        lines = [ln for ln in s.iter_lines() if ln and ln.startswith("data:")]
    assert lines[-1].strip() == "data: [DONE]"
    assert len(lines) > 1
