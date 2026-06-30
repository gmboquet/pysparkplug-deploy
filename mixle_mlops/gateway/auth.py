"""Auth dependencies: resolve a Bearer API key to a user; enforce auth where required."""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException
from sqlmodel import Session

from ..accounts import service
from ..accounts.models import User
from ..config import get_settings
from ..storage.db import get_session


async def current_user(
    authorization: str | None = Header(default=None),
    session: Session = Depends(get_session),
) -> User | None:
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if token:
        resolved = service.resolve_api_key(session, token)
        if resolved is not None:
            return resolved[0]
    if get_settings().require_auth:
        raise HTTPException(status_code=401, detail="missing or invalid API key")
    return None


async def require_user(user: User | None = Depends(current_user)) -> User:
    if user is None:
        raise HTTPException(status_code=401, detail="authentication required")
    return user
