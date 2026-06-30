"""Human-feedback + RLHF routes — the platform differentiator.

  * ``POST /feedback``              — capture a rating / preference / edit.
  * ``POST /rlhf/reward``           — fit the calibrated mixle Bradley-Terry reward model over stored
                                      preferences and return per-item reward WITH uncertainty.
  * ``GET  /rlhf/next-comparison``  — active elicitation: the most-informative next comparison to judge.
  * ``GET  /rlhf/export``           — DPO-style ``{prompt, chosen, rejected}`` JSONL of the preferences.

All routes require an authenticated user (``current_user`` + 401 when auth is required).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from sqlmodel import Session, SQLModel

from ...accounts.models import User
from ...feedback import collect, elicit, loop
from ...feedback.models import Feedback  # noqa: F401  (registers the table in SQLModel.metadata)
from ...feedback.reward import fit_reward
from ...storage.db import get_engine, get_session
from ..auth import require_user

router = APIRouter()


def _ensure_table() -> None:
    """Create the feedback table on demand (idempotent) in case init_db ran before this import."""
    SQLModel.metadata.create_all(get_engine(), tables=[Feedback.__table__])


@router.post("/feedback")
def post_feedback(
    body: dict[str, Any],
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
):
    """Capture one unit of feedback. Body: ``{kind: 'rating'|'preference'|'edit', ...}``."""
    _ensure_table()
    try:
        fb = collect.ingest(session, body, user_id=user.id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"id": fb.id, "kind": fb.kind, "created_at": fb.created_at.isoformat()}


@router.post("/rlhf/reward")
def fit_reward_route(
    body: dict[str, Any] | None = None,
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
):
    """Fit the calibrated Bradley-Terry reward model over stored pairwise preferences."""
    _ensure_table()
    body = body or {}
    model_filter = body.get("model")
    pairs = collect.preference_pairs(session, model=model_filter)
    if len(pairs) < 1:
        raise HTTPException(status_code=422, detail="no pairwise preferences stored yet.")
    try:
        reward = fit_reward(
            pairs,
            pseudo_count=float(body.get("pseudo_count", 0.5)),
            n_boot=int(body.get("n_boot", 400)),
            ci_level=float(body.get("ci_level", 0.9)),
            seed=int(body.get("seed", 0)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return reward.to_dict()


@router.get("/rlhf/next-comparison")
def next_comparison_route(
    model: str | None = Query(default=None),
    n_boot: int = Query(default=200),
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
):
    """Active elicitation: the most-informative next comparison given the current reward posterior."""
    _ensure_table()
    pairs = collect.preference_pairs(session, model=model)
    if len(pairs) < 1:
        raise HTTPException(status_code=422, detail="no pairwise preferences stored yet.")
    try:
        reward = fit_reward(pairs, n_boot=n_boot)
        nxt = elicit.next_comparison(reward)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {
        "item_a": nxt.item_a,
        "item_b": nxt.item_b,
        "expected_information_gain": nxt.score,
        "prob_a_beats_b": nxt.prob_a_beats_b,
        "reward_gap_std": nxt.reward_gap_std,
    }


@router.get("/rlhf/export")
def export_route(
    model: str | None = Query(default=None),
    session: Session = Depends(get_session),
    user: User = Depends(require_user),
):
    """Export stored preferences as DPO-style ``{prompt, chosen, rejected}`` JSONL."""
    _ensure_table()
    jsonl = loop.export_dpo_jsonl(session, model=model)
    return PlainTextResponse(jsonl, media_type="application/x-ndjson")
