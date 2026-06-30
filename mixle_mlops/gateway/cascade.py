"""Cascade router — the FrugalGPT lever (Chen 2023; Ong 2024 RouteLLM) with a tunable quality/cost dial.

Answer with the cheap *local* model when it is confident enough, and escalate to an expensive *frontier* model
only on the hard tail. The confidence is the best-of-N **self-consistency** vote fraction (calibrated, not raw
logprobs) — the defense against the 'wrong-but-confident' failure mode. The ``threshold`` is the dial: it maps to
the loss ratio (cost of a wrong local answer vs. the price of a frontier call) in the Bayes-decision framing —
raise it for more quality (more escalations), lower it for less cost.

The escalate/accept decisions are exactly the in-distribution training signal the self-evolution loop can consume
('the local model was insufficient here'), closing the loop without human labels."""
from __future__ import annotations

from typing import Any

from ..core.adapters import ChatCompletion, ModelAdapter
from .bestofn import best_of_n


async def cascade(local: ModelAdapter, frontier: ModelAdapter, req, *,
                  threshold: float = 0.6, n: int = 5, temperature: float = 0.8) -> tuple[ChatCompletion, dict[str, Any]]:
    """Run the local model (best-of-N) and escalate to ``frontier`` iff its self-consistency confidence is below
    ``threshold``. Returns ``(completion, info)``; ``info`` carries the routing decision for observability + the
    self-evolution training signal."""
    completion, bon = await best_of_n(local, req, n=max(2, int(n)), temperature=temperature)
    confidence = bon.get("confidence")
    escalate = confidence is None or confidence < threshold
    info: dict[str, Any] = {
        "local_model": local.name, "local_confidence": confidence, "threshold": threshold,
        "escalated": escalate, "frontier_model": frontier.name if escalate else None,
    }
    if not escalate:
        return completion, info
    frontier_completion = await frontier.chat(req.model_copy(update={"stream": False}))
    return frontier_completion, info
