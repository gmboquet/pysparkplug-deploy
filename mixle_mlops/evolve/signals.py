"""Usage-signal recording + router self-calibration — the closed-loop payoff of the bridge stack.

Every cascade decision and best-of-N vote is free, in-distribution feedback. ``record_signal`` persists it;
``router_stats`` summarizes the recent window; ``recommend_threshold`` computes the cascade threshold that would
hit a *target* escalation rate — so the router calibrates its own quality/cost dial from real traffic instead of
a hand-tuned constant. (This is calibration on the observed confidence distribution: to escalate a fraction ``t``
of queries, place the threshold at the ``t``-quantile of observed self-consistency confidences.)"""
from __future__ import annotations

import numpy as np
from sqlmodel import Session, select

from .models import SignalRecord


def record_signal(session: Session, model_id: str, *, kind: str,
                  confidence: float | None = None, escalated: bool | None = None) -> SignalRecord:
    rec = SignalRecord(model_id=model_id, kind=kind, confidence=confidence, escalated=escalated)
    session.add(rec)
    session.commit()
    session.refresh(rec)
    return rec


def _recent(session: Session, model_id: str, window: int) -> list[SignalRecord]:
    stmt = (select(SignalRecord).where(SignalRecord.model_id == model_id)
            .order_by(SignalRecord.created_at.desc()).limit(window))
    return list(session.exec(stmt).all())


def router_stats(session: Session, model_id: str, *, window: int = 200) -> dict:
    rows = _recent(session, model_id, window)
    escalations = [r.escalated for r in rows if r.escalated is not None]
    confidences = [r.confidence for r in rows if r.confidence is not None]
    return {
        "model_id": model_id,
        "n": len(rows),
        "escalation_rate": (sum(1 for e in escalations if e) / len(escalations)) if escalations else None,
        "mean_confidence": (float(np.mean(confidences)) if confidences else None),
        "_confidences": confidences,                          # internal: used by recommend_threshold
    }


def recommend_threshold(session: Session, model_id: str, *,
                        target_escalation_rate: float = 0.2, window: int = 200) -> dict:
    stats = router_stats(session, model_id, window=window)
    confidences = stats.pop("_confidences")
    if not confidences:
        return {**stats, "recommended_threshold": None, "reason": "no confidence signal yet"}
    # the threshold that escalates exactly `target` fraction = the target-quantile of observed confidences
    rec = float(np.quantile(np.asarray(confidences, dtype=float), float(target_escalation_rate)))
    return {**stats, "target_escalation_rate": float(target_escalation_rate),
            "recommended_threshold": rec, "n_confidence": len(confidences)}
