"""Integration test of the composed chat pipeline (gateway/routes/chat.py): a single authenticated chat must
flow rate-limit → normalize → RAG → cache → dispatch → persist correctly when the platform features are enabled.
The component packages are unit-tested separately; this proves they are wired together end-to-end."""
import mixle_mlops.storage.db as db
import pytest
from fastapi.testclient import TestClient

from mixle_mlops.cache import reset_cache
from mixle_mlops.config import get_settings
from mixle_mlops.gateway.app import create_app


def _client(tmp_path, monkeypatch, **env):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    get_settings.cache_clear()
    reset_cache()
    db._engine = None
    return TestClient(create_app())


def _signup(c, email):
    raw = c.post("/auth/signup", json={"email": email, "password": "pw12345"}).json()["api_key"]
    return {"Authorization": f"Bearer {raw}"}


def _chat(c, headers, text, **extra):
    body = {"model": "echo", "messages": [{"role": "user", "content": text}]}
    body.update(extra)
    return c.post("/v1/chat/completions", headers=headers, json=body)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    get_settings.cache_clear()
    reset_cache()
    db._engine = None


def test_chat_persists_conversation(tmp_path, monkeypatch):
    """A default authenticated chat records the turn; it shows up in the user's conversation history."""
    with _client(tmp_path, monkeypatch) as c:
        headers = _signup(c, "persist@t.com")
        r = _chat(c, headers, "remember me")
        assert r.status_code == 200
        conv_id = r.headers.get("X-Conversation-Id")          # the route surfaces the id for threading
        assert conv_id
        convs = c.get("/v1/conversations", headers=headers).json()["data"]
        assert len(convs) == 1 and convs[0]["id"] == conv_id
        msgs = c.get(f"/v1/conversations/{conv_id}", headers=headers).json()["messages"]
        roles = [m["role"] for m in msgs]
        assert "user" in roles and "assistant" in roles
        # threading a second turn via the returned id appends, not creates
        assert _chat(c, headers, "again", extra={"conversation_id": conv_id}).status_code == 200
        assert len(c.get("/v1/conversations", headers=headers).json()["data"]) == 1


def test_response_cache_serves_repeats(tmp_path, monkeypatch):
    """With the response cache enabled, an identical second request is a cache hit."""
    with _client(tmp_path, monkeypatch, MIXLE_ENABLE_RESPONSE_CACHE=1) as c:
        headers = _signup(c, "cache@t.com")
        r1 = _chat(c, headers, "same prompt")
        r2 = _chat(c, headers, "same prompt")
        assert r1.status_code == r2.status_code == 200
        assert r1.json()["choices"][0]["message"]["content"] == r2.json()["choices"][0]["message"]["content"]
        stats = c.get("/v1/cache/stats", headers=headers).json()["stats"]
        assert stats.get("hits", 0) >= 1


def test_rate_limit_blocks(tmp_path, monkeypatch):
    """With a 1/min limit, the second request in the window is rejected with 429 + Retry-After."""
    with _client(tmp_path, monkeypatch, MIXLE_RATE_LIMIT_PER_MIN=1) as c:
        headers = _signup(c, "rate@t.com")
        assert _chat(c, headers, "one").status_code == 200
        blocked = _chat(c, headers, "two")
        assert blocked.status_code == 429
        assert "retry-after" in {k.lower() for k in blocked.headers}


def test_rag_flag_is_safe_noop_when_empty(tmp_path, monkeypatch):
    """Opting into RAG with no indexed context must not break the chat (defensive augmentation)."""
    with _client(tmp_path, monkeypatch) as c:
        headers = _signup(c, "rag@t.com")
        r = _chat(c, headers, "hello", extra={"rag": True})
        assert r.status_code == 200
        assert "echo: hello" in r.json()["choices"][0]["message"]["content"]
