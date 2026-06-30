"""Constrained / structured decoding: backend format pass-through, output validation, and the
validate -> repair retry loop (delegating mid-decode masking to the backend, enforcing the shape ourselves)."""
import asyncio

from mixle_mlops.core.adapters import (
    ChatChoice,
    ChatChunkChoice,
    ChatCompletion,
    ChatCompletionChunk,
    ChatMessage,
    ChatRequest,
    ChoiceDelta,
    ModelAdapter,
)
from mixle_mlops.gateway.constrained import (
    constrained_complete,
    to_backend_format,
    validate_output,
)

# a small schema the repaired output must satisfy
SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}


class FlakyJSONAdapter(ModelAdapter):
    """Returns invalid JSON on the first call, then a schema-conforming object on every later call — the
    canonical 'backend ignored guided decoding, repair fixes it' scenario."""

    kind = "llm"

    def __init__(self, name="flaky"):
        self._name = name
        self.calls = 0
        self.last_extra = None

    @property
    def name(self):
        return self._name

    async def chat(self, req: ChatRequest) -> ChatCompletion:
        self.calls += 1
        self.last_extra = dict(req.extra)
        text = "not json at all" if self.calls == 1 else '{"name": "Ada", "age": 36}'
        return ChatCompletion(
            model=req.model,
            choices=[ChatChoice(message=ChatMessage(role="assistant", content=text), finish_reason="stop")],
        )

    async def stream(self, req: ChatRequest):
        completion = await self.chat(req)
        yield ChatCompletionChunk(
            model=req.model,
            choices=[ChatChunkChoice(
                delta=ChoiceDelta(content=completion.choices[0].message.text()), finish_reason="stop")],
        )


def test_to_backend_format_json_schema():
    out = to_backend_format({"json_schema": SCHEMA})
    # OpenAI/vLLM/llama.cpp guided-json key + Ollama `format` key, both emitted.
    assert out["response_format"]["type"] == "json_schema"
    assert out["response_format"]["json_schema"]["schema"] == SCHEMA
    assert out["format"] == SCHEMA


def test_to_backend_format_json_and_grammar():
    assert to_backend_format({"json": True}) == {
        "response_format": {"type": "json_object"},
        "format": "json",
    }
    assert to_backend_format({"grammar": "root ::= object"}) == {"grammar": "root ::= object"}
    assert to_backend_format({}) == {}


def test_validate_output_accepts_conforming():
    ok, parsed, error = validate_output('{"name": "Ada", "age": 36}', {"json_schema": SCHEMA})
    assert ok and error is None
    assert parsed == {"name": "Ada", "age": 36}


def test_validate_output_rejects_missing_required():
    ok, parsed, error = validate_output('{"name": "Ada"}', {"json_schema": SCHEMA})
    assert not ok
    assert error and "age" in error


def test_validate_output_strips_markdown_fence():
    ok, _parsed, error = validate_output(
        '```json\n{"name": "Ada", "age": 36}\n```', {"json_schema": SCHEMA})
    assert ok and error is None


def test_validate_output_rejects_non_json():
    ok, _parsed, error = validate_output("nope", {"json": True})
    assert not ok and error


def test_constrained_complete_repairs_once():
    adapter = FlakyJSONAdapter()
    req = ChatRequest(model="flaky", messages=[ChatMessage(role="user", content="give me a person")])
    completion, info = asyncio.run(
        constrained_complete(adapter, req, {"json_schema": SCHEMA}, max_repairs=2))

    assert info["valid"] is True
    assert info["repairs"] == 1
    assert adapter.calls == 2  # one bad, one repaired
    assert completion.choices[0].message.text() == '{"name": "Ada", "age": 36}'
    # the backend received the forwarded guided-decode keys
    assert adapter.last_extra.get("response_format", {}).get("type") == "json_schema"


def test_constrained_complete_gives_up_without_raising():
    class AlwaysBad(ModelAdapter):
        kind = "llm"

        @property
        def name(self):
            return "bad"

        async def chat(self, req):
            return ChatCompletion(
                model=req.model,
                choices=[ChatChoice(message=ChatMessage(role="assistant", content="still not json"),
                                    finish_reason="stop")])

        async def stream(self, req):
            yield ChatCompletionChunk(model=req.model, choices=[ChatChunkChoice(
                delta=ChoiceDelta(content="still not json"), finish_reason="stop")])

    req = ChatRequest(model="bad", messages=[ChatMessage(role="user", content="x")])
    completion, info = asyncio.run(
        constrained_complete(AlwaysBad(), req, {"json_schema": SCHEMA}, max_repairs=2))
    assert info["valid"] is False
    assert info["repairs"] == 2
    assert completion.choices  # last completion returned, no exception
