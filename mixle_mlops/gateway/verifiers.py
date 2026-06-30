"""Pluggable verifiers for best-of-N selection — beyond self-consistency.

A verifier scores a candidate's quality; ``best_of_n_verified`` samples N and returns the highest-scoring one.
Where a *real* verifier exists, this beats majority voting: Cobbe 2021 — a small generator + a good verifier ≈ a
model ~30× larger. Honest ceiling (Stroebl 2024): the verifier is the bottleneck; an unreliable verifier caps
accuracy and more samples can even hurt — so prefer DETERMINISTIC / checkable verifiers (exact-match, computed
reference) over an LLM judge whenever the task admits one."""
from __future__ import annotations

import re
from typing import Any, Awaitable, Callable

from ..core.adapters import ChatCompletion, ChatMessage, ChatRequest, ModelAdapter
from .bestofn import _sample, _text, extract_answer

Verifier = Callable[[str], Awaitable[float]]      # candidate text -> score (higher is better)

_RATING = re.compile(r"-?\d+(?:\.\d+)?")


def exact_match_verifier(reference: str) -> Verifier:
    """1.0 if the candidate's extracted answer matches ``reference`` (after extraction), else 0.0."""
    ref = extract_answer(reference)

    async def verify(text: str) -> float:
        return 1.0 if extract_answer(text) == ref else 0.0

    return verify


def numeric_verifier(spec: dict[str, Any], *, tol: float = 1e-6) -> Verifier:
    """Compute an exact reference with the program-offload solver, then score candidates by numeric match."""
    from .program_offload import solve_program

    result = solve_program(spec)
    reference = result.get("value", result.get("probability"))

    async def verify(text: str) -> float:
        if reference is None:
            return 0.0
        try:
            candidate = float(extract_answer(text))
        except (TypeError, ValueError):
            return 0.0
        return 1.0 if abs(candidate - float(reference)) <= tol else 0.0

    return verify


def llm_judge_verifier(judge: ModelAdapter, *, criterion: str = "correctness and quality",
                       scale: int = 10) -> Verifier:
    """Ask a judge model to rate a candidate 0..``scale`` for ``criterion``. Use only when no checkable
    verifier exists — an LLM judge is itself fallible (it caps the achievable accuracy)."""
    async def verify(text: str) -> float:
        prompt = (f"Rate the following answer for {criterion} on a scale of 0 to {scale}. "
                  f"Reply with ONLY the number.\n\nAnswer:\n{text}")
        completion = await judge.chat(ChatRequest(
            model=judge.name, messages=[ChatMessage(role="user", content=prompt)], stream=False, temperature=0.0))
        match = _RATING.search(_text(completion))
        return float(match.group(0)) if match else 0.0

    return verify


def build_verifier(spec: dict[str, Any], registry: Any) -> Verifier | None:
    """Construct a verifier from a request spec: {type: exact_match|numeric|llm_judge, ...}."""
    kind = spec.get("type")
    if kind == "exact_match":
        return exact_match_verifier(str(spec.get("reference", "")))
    if kind == "numeric":
        return numeric_verifier(spec.get("spec") or {})
    if kind == "llm_judge":
        model_id = spec.get("model")
        if model_id and registry.has(model_id):
            return llm_judge_verifier(registry.get(model_id),
                                      criterion=str(spec.get("criterion", "correctness and quality")))
    return None


async def best_of_n_verified(adapter: ModelAdapter, req, *, n: int = 5, verifier: Verifier,
                             temperature: float = 0.8) -> tuple[ChatCompletion, dict[str, Any]]:
    """Sample ``n`` candidates and return the one the verifier scores highest. (When mixle.inference.select_best
    is available it can supply conformal optimal-N stopping; this self-contained path takes the argmax.)"""
    n = max(1, int(n))
    candidates = await _sample(adapter, req, n, temperature)
    scores = [await verifier(_text(c)) for c in candidates]
    best = max(range(len(scores)), key=lambda i: scores[i])
    info = {"n": n, "selector": "verifier", "scores": scores,
            "best_score": scores[best], "best_index": best}
    return candidates[best], info
