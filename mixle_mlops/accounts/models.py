"""Identity tables: users and API keys. (Orgs/usage/billing land in a later phase.)"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    email: str = Field(index=True, unique=True)
    hashed_password: str
    is_active: bool = True
    is_admin: bool = False
    created_at: datetime = Field(default_factory=_now)


class ApiKey(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    user_id: str = Field(index=True, foreign_key="user.id")
    name: str = "default"
    prefix: str = Field(index=True)        # leading chars, shown in the UI (the full key is never stored)
    hashed_key: str = Field(index=True)    # sha256(full key)
    kind: str = "api"                       # "api" (programmatic) or "session" (web login)
    created_at: datetime = Field(default_factory=_now)
    last_used: datetime | None = None
    revoked: bool = False
