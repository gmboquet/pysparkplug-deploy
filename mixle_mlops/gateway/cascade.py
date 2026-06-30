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

from ..core.adapters import ChatChoice, ChatCompletion, ChatMessage, ModelAdapter
from .bestofn import best_of_n


async def cascade(local: ModelAdapter, frontier: ModelAdapter, req, *,
                  threshold: float = 0.6, n: int = 5, temperature: float = 0.8) -> tuple[ChatCompletion, dict[str, Any]]:
    """Run the local model and escalate to ``frontier`` on the hard tail. Returns ``(completion, info)``; ``info``
    carries the routing decision for observability + the self-evolution training signal.

    If ``local`` exposes its *own* calibrated escalate signal (``escalation_decision`` -- e.g. a distilled task
    model's conformal/density gate), that drives the route; otherwise the confidence is the best-of-N
    self-consistency vote fraction and ``threshold`` is the quality/cost dial."""
    signal = await local.escalation_decision(req)
    if signal is not None:                                   # the local model knows its own confidence
        escalate = bool(signal["escalate"])
        info = {
            "local_model": local.name, "local_confidence": signal.get("confidence"), "threshold": threshold,
            "escalated": escalate, "frontier_model": frontier.name if escalate else None, "signal": "calibrated",
        }
        if not escalate:
            completion = ChatCompletion(
                model=req.model,
                choices=[ChatChoice(message=ChatMessage(role="assistant", content=str(signal["answer"])))],
            )
            return completion, info
        return await frontier.chat(req.model_copy(update={"stream": False})), info

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
