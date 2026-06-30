"""Generate labeled datasets — from a fitted mixle generative model, or by driving an LLM.

Two sources, one return shape (:class:`GeneratedDataset` — a list of row dicts + an inferred schema):

* **mixle** — :func:`generate_from_mixle` samples a fitted mixle distribution via its sampler
  (``model.sampler(seed).sample(n)``). Because the sampler *is* the data-generating process, every sampled
  value is a ground-truth label by construction — verifiable data, no annotation. Each draw is coerced to a
  row dict; scalar/array/tuple draws get positional column names, dict draws keep their keys.

* **llm** — :func:`generate_from_llm` asks an LLM (any :class:`~mixle_mlops.core.adapters.ModelAdapter`, so
  this works for the OpenAI-compatible backend) to emit ``n`` JSON records matching a caller-supplied schema,
  then parses + validates each record against that schema (coercing types, dropping unparseable rows).

:func:`generate_dataset` is the unified entry point: it takes a :class:`DatasetSpec`, pulls the model from a
:class:`~mixle_mlops.core.registry.ModelRegistry`, and dispatches on ``spec.source``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..core.adapters import ChatMessage, ChatRequest, ModelAdapter


# --------------------------------------------------------------------------------------------------------
# return shapes
# --------------------------------------------------------------------------------------------------------
@dataclass
class GeneratedDataset:
    """A materialisable dataset: ``rows`` of column->value dicts plus an inferred ``schema``."""

    rows: list[dict[str, Any]]
    schema: dict[str, str]
    source: str
    model: str | None = None
    seed: int | None = None
    prompt: str | None = None

    @property
    def n_rows(self) -> int:
        return len(self.rows)


@dataclass
class DatasetSpec:
    """Request for a generated dataset (the route body maps onto this)."""

    source: str                                   # "mixle" | "llm"
    model: str                                    # registry model id
    n: int = 100
    seed: int = 0
    schema: dict[str, str] | None = None          # required for source="llm"
    prompt: str | None = None                     # optional extra instruction for source="llm"
    fmt: str = "jsonl"                             # export format
    columns: list[str] | None = field(default=None)   # optional names for mixle positional columns


# --------------------------------------------------------------------------------------------------------
# schema inference / coercion
# --------------------------------------------------------------------------------------------------------
def _scalar_type(v: Any) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    return "string"


def _coerce_scalar(v: Any) -> Any:
    """Make a sampled value JSON-serialisable (numpy scalar -> python scalar)."""
    if hasattr(v, "item") and not isinstance(v, (list, tuple, dict)):
        try:
            return v.item()
        except Exception:  # pragma: no cover - defensive
            return v
    return v


def _row_from_draw(draw: Any, columns: Sequence[str] | None) -> dict[str, Any]:
    """Coerce one sampled draw into a column->value row dict."""
    if isinstance(draw, dict):
        return {str(k): _coerce_scalar(v) for k, v in draw.items()}
    # array-likes / tuples / lists -> positional columns; scalars -> a single column
    if isinstance(draw, (list, tuple)) or (hasattr(draw, "__len__") and not isinstance(draw, str)
                                           and not hasattr(draw, "item")):
        try:
            values = [_coerce_scalar(x) for x in draw]
        except TypeError:
            values = [_coerce_scalar(draw)]
    else:
        values = [_coerce_scalar(draw)]
    names = list(columns) if columns is not None else [f"x{i}" for i in range(len(values))]
    if len(names) < len(values):                  # pad missing names positionally
        names = names + [f"x{i}" for i in range(len(names), len(values))]
    return {names[i]: values[i] for i in range(len(values))}


def _infer_schema(rows: list[dict[str, Any]]) -> dict[str, str]:
    schema: dict[str, str] = {}
    for row in rows:
        for k, v in row.items():
            if k not in schema:
                schema[k] = _scalar_type(v)
    return schema


# --------------------------------------------------------------------------------------------------------
# mixle source
# --------------------------------------------------------------------------------------------------------
def _model_of(model_or_adapter: Any) -> Any:
    """Accept a fitted mixle distribution, a :class:`MixleAdapter`, or anything exposing ``_model``."""
    if callable(getattr(model_or_adapter, "sampler", None)):
        return model_or_adapter
    inner = getattr(model_or_adapter, "_model", None)
    if inner is not None and callable(getattr(inner, "sampler", None)):
        return inner
    raise ValueError("generate_from_mixle needs a fitted mixle model exposing .sampler(seed).sample(n)")


def generate_from_mixle(
    model_or_adapter: Any,
    n: int,
    seed: int = 0,
    *,
    columns: Sequence[str] | None = None,
    model_id: str | None = None,
) -> GeneratedDataset:
    """Sample ``n`` labeled records from a fitted mixle generative model.

    Uses the model's own sampler (``model.sampler(seed).sample(n)``) — the exact data-generating process,
    so the sampled values are verifiable ground-truth labels. Each draw becomes one row.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    model = _model_of(model_or_adapter)
    sampler = model.sampler(seed=seed)
    draws = sampler.sample(int(n))
    if draws is None:
        draws = []
    # a sampler may return a single object when n==1 and batched=False; normalise to a sequence of n draws
    if not isinstance(draws, (list, tuple)) and not hasattr(draws, "__iter__"):
        draws = [draws]
    rows = [_row_from_draw(d, columns) for d in list(draws)[:n]]
    return GeneratedDataset(
        rows=rows,
        schema=_infer_schema(rows),
        source="mixle",
        model=model_id,
        seed=seed,
    )


# --------------------------------------------------------------------------------------------------------
# llm source
# --------------------------------------------------------------------------------------------------------
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _build_llm_prompt(schema: dict[str, str], n: int, prompt: str | None) -> str:
    fields = ", ".join(f"{k!r} ({t})" for k, t in schema.items())
    parts = [
        f"Generate exactly {n} synthetic data records as a single JSON array.",
        f"Each record is a JSON object with these fields: {fields}.",
        "Respond with ONLY the JSON array, no prose, no code fences.",
    ]
    if prompt:
        parts.insert(0, prompt.strip())
    return "\n".join(parts)


def _coerce_to_schema(rec: dict[str, Any], schema: dict[str, str]) -> dict[str, Any] | None:
    """Coerce/validate one record to the schema; return ``None`` if it cannot be made to fit."""
    if not isinstance(rec, dict):
        return None
    out: dict[str, Any] = {}
    for key, typ in schema.items():
        if key not in rec:
            return None                            # missing required field -> reject the record
        v = rec[key]
        try:
            if typ == "integer":
                out[key] = int(v)
            elif typ == "number":
                out[key] = float(v)
            elif typ == "boolean":
                out[key] = bool(v) if not isinstance(v, str) else v.strip().lower() in ("true", "1", "yes")
            else:
                out[key] = str(v)
        except (TypeError, ValueError):
            return None
    return out


def _parse_llm_records(text: str, schema: dict[str, str]) -> list[dict[str, Any]]:
    """Extract a JSON array from the LLM text and coerce each record to the schema."""
    candidates: list[Any] = []
    stripped = text.strip()
    # strip a ```json ... ``` fence if present
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = re.sub(r"^json\s*", "", stripped, flags=re.IGNORECASE).strip()
    try:
        obj = json.loads(stripped)
        candidates = obj if isinstance(obj, list) else [obj]
    except (ValueError, TypeError):
        m = _JSON_ARRAY_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
                candidates = obj if isinstance(obj, list) else [obj]
            except (ValueError, TypeError):
                candidates = []
    rows: list[dict[str, Any]] = []
    for rec in candidates:
        coerced = _coerce_to_schema(rec, schema)
        if coerced is not None:
            rows.append(coerced)
    return rows


async def generate_from_llm(
    adapter: ModelAdapter,
    schema: dict[str, str],
    n: int,
    prompt: str | None = None,
    *,
    model_id: str | None = None,
    temperature: float | None = None,
) -> GeneratedDataset:
    """Drive an LLM to emit ``n`` JSON records matching ``schema``; parse + validate them.

    The adapter is any :class:`ModelAdapter` (the OpenAI-compatible backend, a hosted LLM, …). Records that
    fail to parse or coerce to the schema are dropped, so the returned dataset is always schema-valid.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if not schema:
        raise ValueError("generate_from_llm requires a non-empty schema")
    user_prompt = _build_llm_prompt(schema, n, prompt)
    req = ChatRequest(
        model=model_id or adapter.name,
        messages=[ChatMessage(role="user", content=user_prompt)],
        temperature=temperature,
        stream=False,
    )
    completion = await adapter.chat(req)
    text = completion.choices[0].message.text() if completion.choices else ""
    rows = _parse_llm_records(text, schema)[:n]
    return GeneratedDataset(
        rows=rows,
        schema=dict(schema),
        source="llm",
        model=model_id or adapter.name,
        prompt=prompt,
    )


# --------------------------------------------------------------------------------------------------------
# unified dispatch
# --------------------------------------------------------------------------------------------------------
async def generate_dataset(spec: DatasetSpec, registry: Any) -> GeneratedDataset:
    """Pull ``spec.model`` from the registry and generate, dispatching on ``spec.source``."""
    if not registry.has(spec.model):
        raise KeyError(f"model {spec.model!r} not found in registry")
    adapter = registry.get(spec.model)
    if spec.source == "mixle":
        return generate_from_mixle(
            adapter, spec.n, spec.seed, columns=spec.columns, model_id=spec.model,
        )
    if spec.source == "llm":
        if not spec.schema:
            raise ValueError("source='llm' requires a schema")
        return await generate_from_llm(
            adapter, spec.schema, spec.n, spec.prompt, model_id=spec.model,
        )
    raise ValueError(f"unknown dataset source {spec.source!r} (expected 'mixle' or 'llm')")
