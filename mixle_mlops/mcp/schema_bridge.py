"""Bridge the platform's MCP tools to OpenAI tool/function declarations.

An MCP tool's ``inputSchema`` and an OpenAI ``function.parameters`` are both JSON-Schema, so the mapping is
structural: name + description carry over, ``inputSchema`` becomes ``parameters``. This is the converter the
tool registry uses to make the platform's own MCP tools callable by a hosted model mid-conversation."""
from __future__ import annotations

from ..core.adapters import FunctionDef, ToolDef
from .server import Tool

_EMPTY_OBJECT = {"type": "object", "properties": {}}


def mcp_schema_to_openai(input_schema: dict | None) -> dict:
    """MCP ``inputSchema`` (JSON-Schema) → OpenAI ``function.parameters`` (JSON-Schema). Structural identity,
    with a valid empty-object default so a paramless tool still validates."""
    if not input_schema:
        return dict(_EMPTY_OBJECT)
    schema = dict(input_schema)
    schema.setdefault("type", "object")
    return schema


def mcp_tool_to_tooldef(tool: Tool) -> ToolDef:
    return ToolDef(function=FunctionDef(
        name=tool.name,
        description=tool.description,
        parameters=mcp_schema_to_openai(tool.input_schema),
    ))
