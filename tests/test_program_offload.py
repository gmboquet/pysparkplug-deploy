"""Program-offload: the exact solver (incl. the safe-eval security lock-down) + the agent offloading a
computation through the mixle_solve tool (PAL end-to-end)."""
import asyncio
import json

import pytest

from mixle_mlops.core.adapters import (
    ChatChoice,
    ChatChunkChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    ChatRequest,
    ChoiceDelta,
    FunctionCall,
    ModelAdapter,
    ToolCall,
)
from mixle_mlops.core.registry import ModelRegistry
from mixle_mlops.gateway.agent_loop import run_agent_loop
from mixle_mlops.gateway.program_offload import safe_eval, solve_program
from mixle_mlops.gateway.tool_registry import ToolRegistry
from mixle_mlops.models import EchoAdapter


def test_safe_eval_computes():
    assert safe_eval("2**10") == 1024
    assert safe_eval("a + b*2", {"a": 3, "b": 4}) == 11
    assert safe_eval("sqrt(16) + max(1, 2)") == 6
    assert abs(safe_eval("pi") - 3.14159) < 1e-4


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')",   # imports
    "(1).__class__",                          # attribute access
    "open('/etc/passwd')",                    # non-whitelisted call
    "[x for x in range(3)]",                  # comprehensions
    "lambda: 1",                              # lambdas
])
def test_safe_eval_blocks_unsafe(expr):
    with pytest.raises((ValueError, SyntaxError)):
        safe_eval(expr)


def test_solve_program_ops():
    assert solve_program({"op": "eval", "expr": "123*456"})["value"] == 56088
    assert abs(solve_program({"op": "normal_prob", "mean": 0, "std": 1, "x": 0})["probability"] - 0.5) < 1e-9
    desc = solve_program({"op": "describe", "data": [1, 2, 3, 4, 5]})
    assert desc["mean"] == 3.0 and desc["median"] == 3.0
    assert "error" in solve_program({"op": "bogus"})


def test_mixle_solve_tool_in_registry():
    reg = ModelRegistry()
    reg.register(EchoAdapter("echo"))
    tools = ToolRegistry(reg, user_id=None)
    assert "mixle_solve" in {t.function.name for t in tools.specs()}
    out = asyncio.run(tools.dispatch("mixle_solve", {"op": "eval", "expr": "6*7"}))
    assert out["value"] == 42


class SolveModel(ModelAdapter):
    """Offloads one multiplication to mixle_solve, then reports the exact result."""
    kind = "llm"
    name = "solver"

    async def chat(self, req):
        tool_msgs = [m for m in req.messages if m.role == "tool"]
        if tool_msgs:
            return ChatCompletion(model=req.model, choices=[ChatChoice(
                message=ChatMessage(role="assistant", content=f"the answer is {tool_msgs[-1].text()}"),
                finish_reason="stop")])
        return ChatCompletion(model=req.model, choices=[ChatChoice(
            message=ChatMessage(role="assistant", content="", tool_calls=[ToolCall(
                function=FunctionCall(name="mixle_solve",
                                      arguments=json.dumps({"op": "eval", "expr": "123*456"})))]),
            finish_reason="tool_calls")])

    async def stream(self, req):
        completion = await self.chat(req)
        yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(
            delta=ChoiceDelta(content=completion.choices[0].message.text()), finish_reason="stop")])


def test_agent_offloads_computation():
    reg = ModelRegistry()
    reg.register(EchoAdapter("echo"))
    tools = ToolRegistry(reg, user_id=None)
    req = ChatRequest(model="solver", messages=[ChatMessage(role="user", content="what is 123*456?")])
    completion = asyncio.run(run_agent_loop(SolveModel(), req, tools, max_iters=3))
    assert "56088" in completion.choices[0].message.text()    # the exact solver result reached the model
