"""Conversation history tables: a ``Conversation`` owns an ordered list of ``Message`` rows.

Follows the ``accounts/models.py`` pattern (uuid pk, utc timestamps). A conversation belongs to a user
and records the model it was held with; each message is a single ``role``/``content`` turn entry.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Conversation(SQLModel, table=True):
    """A persisted chat thread owned by a user."""

    id: str = Field(default_factory=_uuid, primary_key=True)
    user_id: str = Field(index=True)
    title: str = "New conversation"
    model: str | None = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Message(SQLModel, table=True):
    """One turn entry (system/user/assistant/tool) inside a conversation."""

    id: str = Field(default_factory=_uuid, primary_key=True)
    conversation_id: str = Field(index=True, foreign_key="conversation.id")
    role: str = Field(index=True)        # "system" | "user" | "assistant" | "tool"
    content: str = ""
    created_at: datetime = Field(default_factory=_now)
