"""Best-of-N / self-consistency — the test-time-compute lever (Wang 2022 self-consistency; Snell 2024).

Sample N candidates from a (small, local) model and return the consensus answer. With no external verifier the
honest, always-available selector is **self-consistency**: extract each candidate's answer, take the majority,
and report the vote fraction as a *calibrated confidence*. That confidence is itself the signal the cascade
router (FrugalGPT lever) uses to decide whether to escalate to a frontier model.

This buys quality on tasks with a canonical answer (math, multiple-choice, extraction, structured outputs); it
does nothing for open-ended prose, where there is no meaningful 'vote'. The selector is pluggable so a real
verifier (unit tests, a reward model) can replace majority voting where one exists."""
from __future__ import annotations

import asyncio
import re
from collections import Counter
from typing import Any, Callable

from ..core.adapters import ChatCompletion, ModelAdapter

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_LABELLED = re.compile(r"(?:final answer|answer|result)\s*[:=]\s*(.+)", re.IGNORECASE)


def extract_answer(text: str) -> str:
    """A best-effort canonical answer from free-form model text (for voting). Order: \\boxed{} → 'answer: X'
    → trailing number → normalized full text."""
    t = (text or "").strip()
    if not t:
        return ""
    m = _BOXED.search(t)
    if m:
        return m.group(1).strip()
    m = _LABELLED.search(t)
    if m:
        return m.group(1).strip().rstrip(".").strip()
    nums = _NUMBER.findall(t)
    if nums:
        return nums[-1]
    return " ".join(t.lower().split())


async def _sample(adapter: ModelAdapter, req, n: int, temperature: float) -> list[ChatCompletion]:
    sub = req.model_copy(update={"stream": False, "temperature": temperature})
    return list(await asyncio.gather(*[adapter.chat(sub) for _ in range(n)]))


def _text(c: ChatCompletion) -> str:
    return c.choices[0].message.text() if c.choices else ""


async def best_of_n(adapter: ModelAdapter, req, *, n: int = 5, temperature: float = 0.8,
                    selector: str = "self_consistency",
                    extract: Callable[[str], str] | None = None) -> tuple[ChatCompletion, dict[str, Any]]:
    """Sample ``n`` candidates and select one. Returns ``(completion, info)`` where ``info`` carries the
    self-consistency ``confidence`` (winning vote fraction) — the calibrated signal the cascade consumes."""
    n = max(1, int(n))
    candidates = await _sample(adapter, req, n, temperature)
    texts = [_text(c) for c in candidates]
    extract = extract or extract_answer

    if selector == "self_consistency" and n > 1:
        answers = [extract(t) for t in texts]
        counts = Counter(answers)
        winner, votes = counts.most_common(1)[0]
        idx = next(i for i, a in enumerate(answers) if a == winner)
        info = {"n": n, "selector": "self_consistency", "answer": winner,
                "votes": votes, "confidence": votes / n, "distinct": len(counts)}
        return candidates[idx], info

    # no consensus selector (or n==1): return the first candidate, confidence undefined
    return candidates[0], {"n": n, "selector": selector, "confidence": 1.0 if n == 1 else None}
