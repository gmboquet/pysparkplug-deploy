"""Persist + query self-evolution runs (the lineage trail). Thin SQLModel helpers over EvolutionRecord."""
from __future__ import annotations

import json

from sqlmodel import Session, select

from .models import EvolutionRecord
from .worker import EvolutionRun


def record_run(session: Session, run: EvolutionRun, *, user_id: str | None = None) -> EvolutionRecord:
    rec = EvolutionRecord(
        model_id=run.model_id, user_id=user_id, objective=run.objective, operator=run.operator,
        verified=run.verified, promoted=run.promoted, delta=run.delta, n_data=run.n_data,
        verdict_json=json.dumps(run.verdict) if run.verdict else None, error=run.error,
    )
    session.add(rec)
    session.commit()
    session.refresh(rec)
    return rec


def list_runs(session: Session, *, model_id: str | None = None, limit: int = 50) -> list[EvolutionRecord]:
    stmt = select(EvolutionRecord)
    if model_id:
        stmt = stmt.where(EvolutionRecord.model_id == model_id)
    stmt = stmt.order_by(EvolutionRecord.created_at.desc()).limit(limit)
    return list(session.exec(stmt).all())


def get_run(session: Session, run_id: str) -> EvolutionRecord | None:
    return session.get(EvolutionRecord, run_id)
