"""The server-side agentic loop: reason → call tool → observe → repeat, until the model answers or a guard trips.

This is what turns a hosted model + the platform's tools (MCP, RAG, mixle decide/predict) into an *agent*. The
model proposes ``tool_calls``; the loop executes them against the :class:`ToolRegistry`, appends the results as
``role="tool"`` messages, and asks again — exactly the OpenAI function-calling protocol, but with the gateway
(not the client) closing the loop. A ``max_iters`` guard bounds runaway loops; the final call forces
``tool_choice="none"`` so the model must produce a natural-language answer."""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from ..core.adapters import ChatCompletion, ChatMessage, ModelAdapter
from .tool_registry import ToolRegistry

Emit = Callable[[dict[str, Any]], Awaitable[None]]


def _tool_content(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(result, default=str)


async def run_agent_loop(adapter: ModelAdapter, req, tools: ToolRegistry, *,
                         max_iters: int = 6, emit: Emit | None = None) -> ChatCompletion:
    """Run the reason/act loop and return the final completion. ``emit`` (optional) receives tool-step events."""
    specs = tools.specs()
    messages = list(req.messages)
    for _ in range(max_iters):
        sub = req.model_copy(update={
            "messages": messages, "tools": specs, "tool_choice": req.tool_choice or "auto",
            "stream": False, "max_tool_iters": None,
        })
        completion = await adapter.chat(sub)
        choice = completion.choices[0] if completion.choices else None
        if choice is None:
            return completion
        calls = choice.message.tool_calls
        if choice.finish_reason != "tool_calls" or not calls:
            return completion                                  # model produced a final answer
        messages.append(choice.message)                        # the assistant turn that requested the tools
        for call in calls:
            if emit:
                await emit({"type": "tool_call", "id": call.id, "name": call.function.name,
                            "arguments": call.function.arguments})
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = await tools.dispatch(call.function.name, args)
            content = _tool_content(result)
            if emit:
                await emit({"type": "tool_result", "id": call.id, "name": call.function.name, "content": content})
            messages.append(ChatMessage(role="tool", tool_call_id=call.id, name=call.function.name, content=content))
    # iteration budget exhausted — force a tool-free final answer
    final = req.model_copy(update={"messages": messages, "tools": None, "tool_choice": "none",
                                   "stream": False, "max_tool_iters": None})
    return await adapter.chat(final)
