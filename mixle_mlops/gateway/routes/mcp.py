"""Expose the platform's MCP server over HTTP: a single JSON-RPC 2.0 endpoint (``POST /mcp``) carrying
``initialize`` / ``tools/list`` / ``tools/call`` against the app's model registry. This lets remote MCP clients
that prefer HTTP transport (rather than spawning a stdio subprocess) discover and invoke the hosted models.

Auth-gated like the rest of the platform API: a valid Bearer API key is required when ``require_auth`` is on.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from ...accounts.models import User
from ...mcp.server import MCPServer
from ..auth import current_user

router = APIRouter()


@router.post("/mcp")
async def mcp_jsonrpc(message: dict[str, Any], request: Request, user: User | None = Depends(current_user)):
    """Single JSON-RPC 2.0 entrypoint for the MCP server over HTTP."""
    server = MCPServer(request.app.state.registry)
    response = await server.handle(message)
    # JSON-RPC notifications (no id) get a 204-equivalent empty body
    return response if response is not None else {}
