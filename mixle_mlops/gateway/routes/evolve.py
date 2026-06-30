"""``/v1/evolve*`` — drive self-evolution of a hosted mixle model.

  * ``POST /v1/evolve/{model}``          — run one measure→propose→verify→promote loop on supplied data.
  * ``GET  /v1/evolve/{model}/runs``     — the model's evolution lineage (most recent first).
  * ``GET  /v1/evolve/runs/{run_id}``    — one run with its full verdict.
  * ``POST /v1/evolve/{model}/rollback`` — restore the previous champion after a promotion.

Triggering/rolling back mutates a *shared* hosted model, so it is admin-gated; reading lineage needs only auth.
The statistics live entirely in ``mixle.evolve``; this layer only schedules, persists, and guards."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlmodel import Session

from ...accounts.models import User
from ...evolve.lineage import get_run, list_runs, record_run
from ...evolve.policy import EvolutionPolicy
from ...evolve.worker import EvolutionWorker
from ...storage.db import get_session
from ..auth import require_user

router = APIRouter(prefix="/evolve")


class EvolveRequest(BaseModel):
    records: list[Any] = Field(default_factory=list)         # new data to improve on (combined with retained fit_data)
    policy: EvolutionPolicy = Field(default_factory=EvolutionPolicy)
    promote: bool = True                                     # auto-serve a verified win (still gated by policy.approval)


def _require_admin(user: User) -> None:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="self-evolution requires an admin account")


@router.post("/{model_id}")
def trigger(model_id: str, body: EvolveRequest, request: Request,
            user: User = Depends(require_user), session: Session = Depends(get_session)):
    _require_admin(user)
    registry = request.app.state.registry
    if not registry.has(model_id):
        raise HTTPException(status_code=404, detail=f"model {model_id!r} not found")
    run = EvolutionWorker(registry).run(model_id, body.records, body.policy, promote=body.promote)
    rec = record_run(session, run, user_id=user.id)
    return rec.to_dict()


@router.get("/{model_id}/runs")
def runs(model_id: str, user: User = Depends(require_user), session: Session = Depends(get_session)):
    return {"object": "list", "data": [r.to_dict() for r in list_runs(session, model_id=model_id)]}


@router.get("/runs/{run_id}")
def run_detail(run_id: str, user: User = Depends(require_user), session: Session = Depends(get_session)):
    rec = get_run(session, run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="run not found")
    return rec.to_dict()


@router.post("/{model_id}/rollback")
def rollback(model_id: str, request: Request, user: User = Depends(require_user)):
    _require_admin(user)
    registry = request.app.state.registry
    if not registry.has(model_id):
        raise HTTPException(status_code=404, detail=f"model {model_id!r} not found")
    if not EvolutionWorker(registry).rollback(model_id):
        raise HTTPException(status_code=409, detail="nothing to roll back (no promoted predecessor)")
    return {"model_id": model_id, "rolled_back": True}
