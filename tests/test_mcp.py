"""Tests for the MCP server + client.

Drive the JSON-RPC MCP server in-process against a registry holding the echo model:
``initialize`` → ``tools/list`` → ``tools/call('list_models')`` → ``tools/call('chat__echo')``. Then exercise the
HTTP router end-to-end via TestClient (self-contained: app built with create_app(), router included, echo model
registered on app.state.registry, signed-up API key for auth), and the MCPClient over the StdioTransport against a
spawned ``python -m mixle_mlops.mcp.server`` subprocess.

Async coroutines are driven with ``asyncio.run`` so the suite needs no pytest-asyncio plugin.
"""
import asyncio
import json
import sys

import mixle_mlops.storage.db as db
import pytest
from fastapi.testclient import TestClient

from mixle_mlops.config import get_settings
from mixle_mlops.core.registry import ModelRegistry
from mixle_mlops.gateway.app import create_app
from mixle_mlops.gateway.routes import mcp as mcp_route
from mixle_mlops.mcp.client import MCPClient, StdioTransport
from mixle_mlops.mcp.server import MCPServer
from mixle_mlops.models import EchoAdapter


def _registry() -> ModelRegistry:
    reg = ModelRegistry()
    reg.register(EchoAdapter("echo"))
    return reg


# --- in-process server: protocol subset ---
def test_server_initialize_and_list_tools():
    server = MCPServer(_registry())
    init = asyncio.run(server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}))
    assert init["result"]["protocolVersion"]
    assert init["result"]["serverInfo"]["name"] == "mixle-mlops"

    listed = asyncio.run(server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}))
    names = {t["name"] for t in listed["result"]["tools"]}
    assert "list_models" in names
    assert "chat__echo" in names
    chat_tool = next(t for t in listed["result"]["tools"] if t["name"] == "chat__echo")
    assert chat_tool["inputSchema"]["required"] == ["message"]


def test_server_call_list_models():
    server = MCPServer(_registry())
    resp = asyncio.run(server.handle(
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "list_models", "arguments": {}}}
    ))
    result = resp["result"]
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert any(m["id"] == "echo" for m in payload)
    echo = next(m for m in payload if m["id"] == "echo")
    assert echo["kind"] == "llm"
    assert "chat" in echo["capabilities"]


def test_server_call_chat_tool():
    server = MCPServer(_registry())
    resp = asyncio.run(server.handle(
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "chat__echo", "arguments": {"message": "hi there"}}}
    ))
    result = resp["result"]
    assert result["isError"] is False
    assert "echo: hi there" in result["content"][0]["text"]


def test_server_unknown_tool_and_method():
    server = MCPServer(_registry())
    bad_tool = asyncio.run(server.handle(
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
         "params": {"name": "nope", "arguments": {}}}
    ))
    assert bad_tool["error"]["code"] == -32602
    bad_method = asyncio.run(server.handle({"jsonrpc": "2.0", "id": 6, "method": "frobnicate", "params": {}}))
    assert bad_method["error"]["code"] == -32601


def test_server_notification_returns_none():
    server = MCPServer(_registry())
    out = asyncio.run(server.handle({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}))
    assert out is None


# --- HTTP router, end-to-end and self-contained ---
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    db._engine = None
    app = create_app()
    app.include_router(mcp_route.router, tags=["mcp"])   # not depending on app.py edits
    with TestClient(app) as c:
        if not c.app.state.registry.has("echo"):
            c.app.state.registry.register(EchoAdapter("echo"))
        yield c
    get_settings.cache_clear()
    db._engine = None


def test_http_mcp_endpoint(client):
    raw = client.post("/auth/signup", json={"email": "mcp@t.com", "password": "pw12345"}).json()["api_key"]
    headers = {"Authorization": f"Bearer {raw}"}

    init = client.post("/mcp", headers=headers,
                       json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init.status_code == 200
    assert init.json()["result"]["serverInfo"]["name"] == "mixle-mlops"

    listed = client.post("/mcp", headers=headers,
                         json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    names = {t["name"] for t in listed.json()["result"]["tools"]}
    assert {"list_models", "chat__echo"} <= names

    called = client.post("/mcp", headers=headers,
                         json={"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                               "params": {"name": "list_models", "arguments": {}}})
    payload = json.loads(called.json()["result"]["content"][0]["text"])
    assert any(m["id"] == "echo" for m in payload)


def test_http_mcp_requires_auth(client):
    if get_settings().require_auth:
        r = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        assert r.status_code == 401


# --- MCPClient over a spawned stdio server subprocess ---
def test_client_over_stdio_subprocess():
    async def _run():
        transport = StdioTransport([sys.executable, "-m", "mixle_mlops.mcp.server"])
        async with MCPClient(transport) as cli:
            await cli.initialize()
            tools = await cli.list_tools()
            names = {t["name"] for t in tools}
            assert "list_models" in names
            assert "chat__echo" in names

            text = await cli.call_tool_text("list_models", {})
            payload = json.loads(text)
            assert any(m["id"] == "echo" for m in payload)

            reply = await cli.call_tool_text("chat__echo", {"message": "ping"})
            assert "echo: ping" in reply

    asyncio.run(_run())
