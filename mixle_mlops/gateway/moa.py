"""Mixture-of-Agents (Wang 2024) — several proposer models answer, an aggregator synthesizes one response.

Layered aggregation of open models can rival a single large model on hard tasks, *when the proposers are
individually competent and diverse*. The honest caveats (enforced by the caller, not hidden): a homogeneous
fleet of one small model captures little MoA benefit — the lever pays when you can field 2+ distinct/competent
proposers (or pull in an occasional stronger one) and let the local model aggregate. Error-correlated proposers
hurt; selecting *which* agents enter the mix (focal-diversity pruning) is the mixle-native refinement
(``mixle.stats`` ranking / mixture responsibilities) and is left as a follow-up — this builds the core orchestration."""
from __future__ import annotations

import asyncio
from typing import Any

from ..core.adapters import ChatCompletion, ChatMessage, ModelAdapter

AGGREGATE_INSTRUCTION = (
    "You have been given candidate responses from several models to the user's latest query. "
    "Synthesize a single, high-quality answer: critically evaluate them, reconcile disagreements, correct errors, "
    "and do not copy any one verbatim. Output only the final answer.\n\nCandidate responses:\n"
)


def _text(c: ChatCompletion) -> str:
    return c.choices[0].message.text() if c.choices else ""


def _aggregate_prompt(proposals: list[str]) -> str:
    body = "\n".join(f"[{i + 1}] {t.strip()}" for i, t in enumerate(proposals))
    return AGGREGATE_INSTRUCTION + body


async def _layer(proposers: list[ModelAdapter], req, temperature: float) -> list[str]:
    sub = req.model_copy(update={"stream": False, "temperature": temperature})
    results = await asyncio.gather(*[p.chat(sub) for p in proposers])
    return [_text(c) for c in results]


async def mixture_of_agents(proposers: list[ModelAdapter], aggregator: ModelAdapter, req, *,
                            layers: int = 1, temperature: float = 0.7) -> tuple[ChatCompletion, dict[str, Any]]:
    """Run ``layers`` proposer rounds (each round sees the previous round's proposals) then a final aggregation.
    Returns ``(final_completion, info)``."""
    proposals: list[str] = []
    base = req
    for _ in range(max(1, int(layers))):
        proposals = await _layer(proposers, base, temperature)
        # next round's proposers see this round's proposals as additional context (the MoA refinement loop)
        base = req.model_copy(update={"messages": list(req.messages) + [
            ChatMessage(role="user", content=_aggregate_prompt(proposals))]})

    agg_messages = list(req.messages) + [ChatMessage(role="user", content=_aggregate_prompt(proposals))]
    final = await aggregator.chat(req.model_copy(update={"messages": agg_messages, "stream": False}))
    info = {"proposers": [p.name for p in proposers], "aggregator": aggregator.name,
            "layers": max(1, int(layers)), "n_proposals": len(proposals)}
    return final, info
