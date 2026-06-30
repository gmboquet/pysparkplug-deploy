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
    hashed_password: str | None = None        # None for OAuth-only accounts (Google / Apple)
    is_active: bool = True
    is_admin: bool = False
    created_at: datetime = Field(default_factory=_now)


class OAuthIdentity(SQLModel, table=True):
    """Links a User to an external OIDC identity (one row per provider the user has connected)."""

    id: str = Field(default_factory=_uuid, primary_key=True)
    user_id: str = Field(index=True, foreign_key="user.id")
    provider: str = Field(index=True)         # "google" | "apple"
    subject: str = Field(index=True)          # the provider's stable user id (the id_token `sub`)
    email: str | None = None
    created_at: datetime = Field(default_factory=_now)


class DeviceCode(SQLModel, table=True):
    """An RFC 8628 device-authorization grant, used so the CLI agent can log in via a browser."""

    id: str = Field(default_factory=_uuid, primary_key=True)
    device_code_hash: str = Field(index=True)  # sha256(device_code); the raw code is shown only to the CLI
    user_code: str = Field(index=True)         # short human-entered code, e.g. "WDJB-MJHT"
    status: str = "pending"                     # pending | approved | denied | claimed | expired
    user_id: str | None = Field(default=None, foreign_key="user.id")
    created_at: datetime = Field(default_factory=_now)
    expires_at: datetime
    interval: int = 5                           # min seconds between polls


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
