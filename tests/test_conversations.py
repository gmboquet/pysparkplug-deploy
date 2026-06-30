"""Conversation persistence + export: create a conversation, append turns via ``persist_turn``,
list/get over HTTP, and export json + markdown (pdf when reportlab is present).

Self-contained: builds the app via ``create_app()``, includes the conversations router, signs up for
a key, and drives everything over HTTP through the TestClient — no dependence on app.py edits.
"""

import importlib.util
import json

import mixle_mlops.storage.db as db
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from mixle_mlops.config import get_settings
from mixle_mlops.conversations import service
from mixle_mlops.conversations.export import ExportError, export_conversation
from mixle_mlops.conversations.models import Conversation, Message  # noqa: F401
from mixle_mlops.gateway.app import create_app
from mixle_mlops.gateway.routes import conversations as conv_routes

_HAS_REPORTLAB = importlib.util.find_spec("reportlab") is not None


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    app = create_app()
    app.include_router(conv_routes.router, prefix="/v1", tags=["conversations"])
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()
    db._engine = None


def _signup(client, email="conv@t.com"):
    raw = client.post("/auth/signup", json={"email": email, "password": "pw12345"}).json()["api_key"]
    return {"Authorization": f"Bearer {raw}"}


def _seed_turns(client, headers):
    """Persist two turns into a fresh conversation directly via the service (the chat.py seam)."""
    conv_routes._ensure_table()
    with Session(db.get_engine()) as session:
        # use persist_turn the way the chat route will
        from mixle_mlops.accounts import service as acct_service
        # resolve the signed-up user id from the api key header
        token = headers["Authorization"].split(" ", 1)[1]
        resolved = acct_service.resolve_api_key(session, token)
        assert resolved is not None
        uid = resolved[0].id
        conv = service.persist_turn(session, uid, None, "Hello there", "Hi! How can I help?", model="echo")
        service.persist_turn(session, uid, conv.id, "What is 2+2?", "4", model="echo")
        return conv.id


# ---------- unit-level: service + export ----------

def test_persist_turn_creates_and_threads(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    conv_routes._ensure_table()
    with Session(db.get_engine()) as session:
        conv = service.persist_turn(session, "u1", None, "first message please", "ok", model="echo")
        assert conv.title.startswith("first message")
        assert conv.model == "echo"
        same = service.persist_turn(session, "u1", conv.id, "second", "ok2")
        assert same.id == conv.id  # threaded into the same conversation
        _, messages = service.get_conversation(session, conv.id)
        assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]
        assert messages[0].content == "first message please"
    get_settings.cache_clear()
    db._engine = None


def test_export_json_and_markdown_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    conv_routes._ensure_table()
    with Session(db.get_engine()) as session:
        conv = service.persist_turn(session, "u9", None, "Q one", "A one", model="echo")
        result = service.get_conversation(session, conv.id)
        assert result is not None
        c, msgs = result

        data, media, suffix = export_conversation(c, msgs, "json")
        assert media == "application/json" and suffix == "json"
        obj = json.loads(data)
        assert obj["title"] == conv.title
        assert [m["role"] for m in obj["messages"]] == ["user", "assistant"]

        data, media, suffix = export_conversation(c, msgs, "markdown")
        assert media == "text/markdown" and suffix == "md"
        text = data.decode()
        assert "Q one" in text and "A one" in text and "## User" in text

        with pytest.raises(ExportError):
            export_conversation(c, msgs, "xml")
    get_settings.cache_clear()
    db._engine = None


# ---------- end-to-end over HTTP ----------

def test_list_and_get_over_http(client):
    headers = _signup(client)
    conv_id = _seed_turns(client, headers)

    listed = client.get("/v1/conversations", headers=headers).json()
    assert listed["object"] == "list"
    assert any(c["id"] == conv_id for c in listed["data"])

    got = client.get(f"/v1/conversations/{conv_id}", headers=headers).json()
    assert got["id"] == conv_id
    assert [m["role"] for m in got["messages"]] == ["user", "assistant", "user", "assistant"]
    assert got["messages"][0]["content"] == "Hello there"


def test_export_routes_over_http(client):
    headers = _signup(client, email="exp@t.com")
    conv_id = _seed_turns(client, headers)

    r = client.get(f"/v1/conversations/{conv_id}/export", headers=headers, params={"format": "json"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    obj = json.loads(r.content)
    assert obj["id"] == conv_id

    r = client.get(f"/v1/conversations/{conv_id}/export", headers=headers, params={"format": "markdown"})
    assert r.status_code == 200
    assert "## Assistant" in r.text

    # unknown format → 422
    r = client.get(f"/v1/conversations/{conv_id}/export", headers=headers, params={"format": "docx"})
    assert r.status_code == 422


@pytest.mark.skipif(not _HAS_REPORTLAB, reason="reportlab not installed (export extra)")
def test_export_pdf_over_http(client):
    headers = _signup(client, email="pdf@t.com")
    conv_id = _seed_turns(client, headers)
    r = client.get(f"/v1/conversations/{conv_id}/export", headers=headers, params={"format": "pdf"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/pdf")
    assert r.content[:4] == b"%PDF"


def test_delete_and_isolation(client):
    headers_a = _signup(client, email="a@t.com")
    headers_b = _signup(client, email="b@t.com")
    conv_id = _seed_turns(client, headers_a)

    # user B cannot see or fetch user A's conversation
    assert all(c["id"] != conv_id for c in client.get("/v1/conversations", headers=headers_b).json()["data"])
    assert client.get(f"/v1/conversations/{conv_id}", headers=headers_b).status_code == 404
    assert client.delete(f"/v1/conversations/{conv_id}", headers=headers_b).status_code == 404

    # owner deletes
    assert client.delete(f"/v1/conversations/{conv_id}", headers=headers_a).status_code == 200
    assert client.get(f"/v1/conversations/{conv_id}", headers=headers_a).status_code == 404


def test_requires_auth(client):
    assert client.get("/v1/conversations").status_code == 401
    assert client.get("/v1/conversations/x/export").status_code == 401
    assert client.delete("/v1/conversations/x").status_code == 401
