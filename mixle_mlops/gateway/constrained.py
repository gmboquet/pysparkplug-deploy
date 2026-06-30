"""Constrained / structured decoding — the "the model must return THIS shape" lever.

Honest framing: true on-the-fly grammar-masked sampling happens INSIDE the model's decode loop (mask the
logits at every step to forbid tokens that would violate the grammar). Through an OpenAI-compatible HTTP API we
do not have the logits, so we cannot mask in-process. What we *can* do — and what this module does — is the two
halves the proxy actually controls:

  1. PASS-THROUGH the constraint to a backend that supports guided decoding. Ollama takes ``format`` (a JSON
     schema or the literal ``"json"``); vLLM / llama.cpp / TGI take ``response_format`` (OpenAI's
     ``{"type": "json_schema", ...}``) or a raw ``grammar`` (GBNF/Lark). The actual mid-decode masking is
     delegated to the backend. We emit the right ``extra`` keys and the proxy forwards them verbatim
     (``OpenAICompatAdapter`` does ``body.update(req.extra)``).

  2. VALIDATE the returned text against the spec and REPAIR via a bounded retry loop — a backend-agnostic
     guarantee that holds even when the backend ignores or lacks guided decoding. This is the part that makes
     the contract real for every backend, including ``EchoAdapter``-style models with no guided-decode support.

So: pass-through + validate + repair. We never claim to mask logits in-process.

JSON-schema validation uses ``jsonschema`` when importable (full Draft-2020 semantics) and otherwise a
minimal, dependency-light structural check (``json.loads`` + ``required`` keys + top-level ``type``/property
types). Grammar validation: mixle's ``HeterogeneousPCFGDistribution`` is a *probabilistic* CNF PCFG over
sequences of distribution-emitted tokens — it has no parser that ingests a textual grammar string (GBNF/Lark)
and decides character-level membership, so it cannot validate an arbitrary ``grammar`` spec here. We say so
honestly and fall back to a "valid JSON" check for grammar specs (the common case is a JSON-shaped grammar);
masking for a real grammar is delegated to the backend.
"""
from __future__ import annotations

import json
from typing import Any

from ..core.adapters import ChatCompletion, ChatRequest, ModelAdapter

# Spec shapes accepted (any one):
#   {"json_schema": {...}}          -> JSON validated against the schema (+ forwarded as response_format/format)
#   {"json": true}                  -> any valid JSON object (+ forwarded as the "json" format)
#   {"grammar": "<gbnf/lark>"}      -> forwarded as a backend grammar; locally validated as "valid JSON" (see above)
Spec = dict[str, Any]

_REPAIR_TEMPLATE = (
    "Your last output was invalid: {error}. "
    "Return ONLY output matching the schema, with no prose, no markdown fences, no explanation."
)


# --------------------------------------------------------------------------------------------------------------
# 1. backend pass-through
# --------------------------------------------------------------------------------------------------------------
def to_backend_format(spec: Spec) -> dict[str, Any]:
    """Translate a constraint ``spec`` into the ``extra`` keys backends understand for guided decoding.

    Returns a dict to merge into ``req.extra`` so ``OpenAICompatAdapter`` forwards it. We emit *both* the
    OpenAI ``response_format`` (vLLM / llama.cpp / TGI / OpenAI) and Ollama's ``format`` so a single request
    works against either family; a backend ignores the key it does not recognize.
    """
    if not isinstance(spec, dict):
        return {}

    if spec.get("json_schema") is not None:
        schema = spec["json_schema"]
        name = str(spec.get("name") or schema.get("title") or "response")
        return {
            # OpenAI / vLLM / llama.cpp / TGI guided-json
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            },
            # Ollama takes the bare schema in `format`
            "format": schema,
        }

    if spec.get("grammar") is not None:
        # vLLM / llama.cpp accept a raw grammar (GBNF / Lark). Masking is the backend's job.
        return {"grammar": spec["grammar"]}

    if spec.get("json"):
        return {"response_format": {"type": "json_object"}, "format": "json"}

    return {}


# --------------------------------------------------------------------------------------------------------------
# 2. output validation
# --------------------------------------------------------------------------------------------------------------
def _strip_fences(text: str) -> str:
    """Drop a leading/trailing ```... ``` markdown fence if the model wrapped its JSON in one."""
    t = (text or "").strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1:]
        if t.endswith("```"):
            t = t[: -3]
    return t.strip()


_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def _type_mismatch(value: Any, json_type: str) -> bool:
    """True if ``value`` does not satisfy a JSON-Schema ``type`` token. Treats ``bool`` as distinct from
    number/integer (JSON booleans are not numbers), matching ``jsonschema`` semantics."""
    if json_type not in _JSON_TYPES:
        return False  # unknown/composite type token → don't enforce
    if json_type in ("number", "integer") and isinstance(value, bool):
        return True
    return not isinstance(value, _JSON_TYPES[json_type])


def _minimal_schema_check(obj: Any, schema: dict[str, Any]) -> str | None:
    """A tiny, dependency-light structural check: top-level ``type``, ``required`` keys, and declared property
    types one level deep. Returns an error string, or ``None`` if it passes. Used only when ``jsonschema`` is
    not importable."""
    if not isinstance(schema, dict):
        return None
    exp = schema.get("type")
    if isinstance(exp, str) and _type_mismatch(obj, exp):
        return f"expected type {exp!r}, got {type(obj).__name__}"
    if isinstance(obj, dict):
        for key in schema.get("required", []) or []:
            if key not in obj:
                return f"missing required field {key!r}"
        props = schema.get("properties", {}) or {}
        for key, sub in props.items():
            if key in obj and isinstance(sub, dict):
                t = sub.get("type")
                if isinstance(t, str) and _type_mismatch(obj[key], t):
                    return f"field {key!r} expected type {t!r}, got {type(obj[key]).__name__}"
    return None


def validate_output(text: str, spec: Spec) -> tuple[bool, Any, str | None]:
    """Validate model ``text`` against ``spec``.

    Returns ``(ok, parsed, error)``: ``parsed`` is the decoded JSON object (or ``None``); ``error`` is a short
    human-readable reason on failure (fed into the repair instruction).
    """
    if not isinstance(spec, dict):
        return True, None, None

    candidate = _strip_fences(text)

    # ---- JSON schema ----
    if spec.get("json_schema") is not None:
        schema = spec["json_schema"]
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError) as exc:
            return False, None, f"not valid JSON ({exc})"
        try:
            import jsonschema  # full Draft semantics when available

            try:
                jsonschema.validate(parsed, schema)
            except jsonschema.ValidationError as exc:
                # exc.message is concise; the full exc is verbose
                return False, parsed, f"schema violation: {exc.message}"
            return True, parsed, None
        except ImportError:
            err = _minimal_schema_check(parsed, schema)
            return (err is None), parsed, err

    # ---- grammar: mixle's PCFG cannot parse a textual grammar string (see module docstring), so the local
    #      check is "is it valid JSON"; real grammar enforcement is delegated to the backend's masker. ----
    if spec.get("grammar") is not None:
        try:
            parsed = json.loads(candidate)
            return True, parsed, None
        except (json.JSONDecodeError, TypeError) as exc:
            return False, None, f"not valid JSON for grammar fallback ({exc})"

    # ---- any valid JSON ----
    if spec.get("json"):
        try:
            parsed = json.loads(candidate)
            return True, parsed, None
        except (json.JSONDecodeError, TypeError) as exc:
            return False, None, f"not valid JSON ({exc})"

    # unknown / empty spec → nothing to enforce
    return True, None, None


# --------------------------------------------------------------------------------------------------------------
# 3. pass-through + validate + repair
# --------------------------------------------------------------------------------------------------------------
def _completion_text(c: ChatCompletion) -> str:
    return c.choices[0].message.text() if c.choices else ""


async def constrained_complete(
    adapter: ModelAdapter,
    req: ChatRequest,
    spec: Spec,
    *,
    max_repairs: int = 2,
) -> tuple[ChatCompletion, dict[str, Any]]:
    """Run a non-streaming completion under a constraint ``spec``: forward guided-decode keys to the backend,
    validate the result, and repair via a bounded retry loop.

    Returns ``(completion, info)`` where ``info = {valid, repairs, schema}``. If the output is still invalid
    after ``max_repairs`` retries, the *last* completion is returned with ``valid=False`` — this never raises
    into the chat path.
    """
    # forward the constraint to a guided-decoding backend (no-op for backends that ignore it).
    base_extra = dict(req.extra)
    base_extra.update(to_backend_format(spec))
    work = req.model_copy(update={"stream": False, "extra": base_extra})

    completion = await adapter.chat(work)
    ok, _parsed, error = validate_output(_completion_text(completion), spec)

    repairs = 0
    while not ok and repairs < max_repairs:
        repairs += 1
        # append the invalid output + a repair instruction, then re-ask.
        from ..core.adapters import ChatMessage

        repair_messages = list(work.messages) + [
            ChatMessage(role="assistant", content=_completion_text(completion)),
            ChatMessage(role="user", content=_REPAIR_TEMPLATE.format(error=error or "did not match the schema")),
        ]
        work = work.model_copy(update={"messages": repair_messages})
        completion = await adapter.chat(work)
        ok, _parsed, error = validate_output(_completion_text(completion), spec)

    info = {
        "valid": ok,
        "repairs": repairs,
        "schema": spec.get("json_schema") or ("grammar" if spec.get("grammar") is not None else spec.get("json")),
    }
    if not ok and error:
        info["error"] = error
    return completion, info
