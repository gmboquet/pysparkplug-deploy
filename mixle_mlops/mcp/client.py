"""A small MCP **client** so hosted models (and platform code) can call *external* MCP tools.

Minimal JSON-RPC 2.0 over two transports:

* :class:`StdioTransport` — spawn an MCP server subprocess and talk newline-delimited JSON-RPC over its stdio
  (the common local transport, e.g. ``npx some-mcp-server``).
* :class:`HTTPTransport` — POST JSON-RPC to an MCP server's HTTP endpoint.

Usage::

    async with MCPClient(StdioTransport(["python", "-m", "mixle_mlops.mcp.server"])) as cli:
        await cli.initialize()
        tools = await cli.list_tools()
        out = await cli.call_tool("list_models", {})

No third-party dependency for stdio; HTTP uses ``httpx`` if importable (already a FastAPI/test dep) else
``urllib`` in a thread. If the official ``mcp`` package is installed, prefer its richer client for production —
this is the dependency-light fallback that always works.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import Any


class MCPClientError(Exception):
    pass


class Transport(ABC):
    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def request(self, message: dict[str, Any]) -> dict[str, Any] | None: ...

    @abstractmethod
    async def notify(self, message: dict[str, Any]) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...


class StdioTransport(Transport):
    """Spawn an MCP server subprocess; exchange newline-delimited JSON-RPC over its stdio."""

    def __init__(self, command: list[str], *, cwd: str | None = None, env: dict[str, str] | None = None):
        self.command = command
        self.cwd = cwd
        self.env = env
        self._proc: subprocess.Popen | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
            cwd=self.cwd,
            env=self.env,
        )

    def _proc_or_raise(self) -> subprocess.Popen:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise MCPClientError("transport not started")
        return self._proc

    async def _send(self, message: dict[str, Any]) -> None:
        proc = self._proc_or_raise()
        loop = asyncio.get_event_loop()

        def _write():
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()

        await loop.run_in_executor(None, _write)

    async def _read(self) -> dict[str, Any]:
        proc = self._proc_or_raise()
        loop = asyncio.get_event_loop()
        while True:
            line = await loop.run_in_executor(None, proc.stdout.readline)
            if not line:
                raise MCPClientError("MCP server closed the connection")
            line = line.strip()
            if not line:
                continue
            return json.loads(line)

    async def request(self, message: dict[str, Any]) -> dict[str, Any] | None:
        async with self._lock:
            await self._send(message)
            return await self._read()

    async def notify(self, message: dict[str, Any]) -> None:
        async with self._lock:
            await self._send(message)

    async def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None


class HTTPTransport(Transport):
    """POST JSON-RPC to an MCP server's HTTP endpoint."""

    def __init__(self, url: str, *, headers: dict[str, str] | None = None, timeout: float = 30.0):
        self.url = url
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.timeout = timeout

    async def start(self) -> None:
        return None

    async def request(self, message: dict[str, Any]) -> dict[str, Any] | None:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=self.timeout) as cli:
                resp = await cli.post(self.url, json=message, headers=self.headers)
                resp.raise_for_status()
                if not resp.content:
                    return None
                return resp.json()
        except ImportError:
            return await self._urllib_request(message)

    async def _urllib_request(self, message: dict[str, Any]) -> dict[str, Any] | None:
        import urllib.request

        loop = asyncio.get_event_loop()

        def _do() -> dict[str, Any] | None:
            data = json.dumps(message).encode()
            req = urllib.request.Request(self.url, data=data, headers=self.headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310
                body = r.read()
            return json.loads(body) if body else None

        return await loop.run_in_executor(None, _do)

    async def notify(self, message: dict[str, Any]) -> None:
        await self.request(message)

    async def close(self) -> None:
        return None


class MCPClient:
    """A minimal MCP client: ``initialize`` → ``list_tools`` → ``call_tool``, over any :class:`Transport`."""

    def __init__(self, transport: Transport):
        self.transport = transport
        self._id = 0

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    async def connect(self) -> None:
        await self.transport.start()

    async def close(self) -> None:
        await self.transport.close()

    async def _call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        msg = {"jsonrpc": "2.0", "id": self._next_id(), "method": method, "params": params or {}}
        resp = await self.transport.request(msg)
        if resp is None:
            raise MCPClientError(f"no response for {method!r}")
        if "error" in resp:
            err = resp["error"]
            raise MCPClientError(f"{method} failed [{err.get('code')}]: {err.get('message')}")
        return resp.get("result")

    async def initialize(self, *, client_name: str = "mixle-mlops-client", client_version: str = "0.1.0") -> dict:
        result = await self._call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": client_name, "version": client_version},
            },
        )
        # MCP handshake: the client confirms with an initialized notification
        await self.transport.notify({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        return result

    async def list_tools(self) -> list[dict[str, Any]]:
        result = await self._call("tools/list")
        return result.get("tools", [])

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call a tool; returns the raw MCP result ``{content: [...], isError: bool}``."""
        return await self._call("tools/call", {"name": name, "arguments": arguments or {}})

    async def call_tool_text(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Convenience: call a tool and concatenate its text content parts."""
        result = await self.call_tool(name, arguments)
        if result.get("isError"):
            text = _content_text(result)
            raise MCPClientError(f"tool {name!r} errored: {text}")
        return _content_text(result)


def _content_text(result: dict[str, Any]) -> str:
    parts = result.get("content") or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text")
