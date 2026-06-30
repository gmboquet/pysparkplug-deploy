"""The lineage table: one row per self-evolution run (what was tried, what won, whether it was promoted)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EvolutionRecord(SQLModel, table=True):
    __tablename__ = "evolution_record"

    id: str = Field(default_factory=_uuid, primary_key=True)
    model_id: str = Field(index=True)
    user_id: str | None = Field(default=None, index=True)
    objective: str = "nll"
    operator: str | None = None
    verified: bool = False
    promoted: bool = False
    delta: float = 0.0
    n_data: int = 0
    verdict_json: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)

    def to_dict(self) -> dict:
        import json

        return {
            "id": self.id,
            "model_id": self.model_id,
            "objective": self.objective,
            "operator": self.operator,
            "verified": self.verified,
            "promoted": self.promoted,
            "delta": self.delta,
            "n_data": self.n_data,
            "verdict": json.loads(self.verdict_json) if self.verdict_json else None,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
        }
