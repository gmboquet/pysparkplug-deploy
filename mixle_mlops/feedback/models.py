"""Feedback persistence table — the raw human signal the RLHF loop learns from.

A single ``Feedback`` row captures one of three kinds of signal, following the ``accounts/models.py``
pattern (uuid pk, utc timestamps):

  * ``kind="rating"``     — a 👍/👎 (or scalar) rating of one message; ``value`` holds it,
    ``message_id`` the rated message.
  * ``kind="preference"`` — a pairwise comparison: ``chosen_id`` beat ``rejected_id`` (item ids are
    response/message ids or model ids — the reward model treats them as opaque item labels).
  * ``kind="edit"``       — the user rewrote a response; the edited text lives in ``payload``.

``payload`` is a free-form JSON blob (stored as a JSON string) for anything kind-specific: the prompt,
the two candidate texts behind a preference, the edited text, latency, etc.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Field, SQLModel


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Feedback(SQLModel, table=True):
    """One unit of captured human feedback (rating / preference / edit)."""

    id: str = Field(default_factory=_uuid, primary_key=True)
    user_id: str | None = Field(default=None, index=True)        # who gave the feedback (nullable for anon)
    conversation_id: str | None = Field(default=None, index=True)
    message_id: str | None = Field(default=None, index=True)     # the rated/edited message
    model: str | None = Field(default=None, index=True)          # model that produced the message
    kind: str = Field(index=True)                                # "rating" | "preference" | "edit"
    value: float | None = None                                   # rating: +1 / -1 / scalar
    chosen_id: str | None = Field(default=None, index=True)      # preference: the winning item id
    rejected_id: str | None = Field(default=None, index=True)    # preference: the losing item id
    payload: str | None = None                                   # JSON-encoded extra context
    created_at: datetime = Field(default_factory=_now)

    # --- convenience helpers (not columns) ---
    def payload_dict(self) -> dict[str, Any]:
        if not self.payload:
            return {}
        try:
            obj = json.loads(self.payload)
            return obj if isinstance(obj, dict) else {"value": obj}
        except (ValueError, TypeError):
            return {}

    @staticmethod
    def encode_payload(payload: dict[str, Any] | None) -> str | None:
        return None if payload is None else json.dumps(payload, default=str)
