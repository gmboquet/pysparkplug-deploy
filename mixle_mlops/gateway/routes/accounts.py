"""Identity routes: signup, login, and API-key management (used by the chat UI and by API clients)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from ...accounts import service
from ...accounts.models import User
from ...accounts.service import AccountError
from ...storage.db import get_session
from ..auth import require_user

router = APIRouter()


class Credentials(BaseModel):
    email: str
    password: str


class KeyRequest(BaseModel):
    name: str = "default"


def _user_public(user: User) -> dict:
    return {"id": user.id, "email": user.email, "is_admin": user.is_admin}


@router.post("/auth/signup")
def signup(body: Credentials, session: Session = Depends(get_session)):
    try:
        user = service.create_user(session, body.email, body.password)
    except AccountError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _key, raw = service.create_api_key(session, user, name="default")
    return {"user": _user_public(user), "api_key": raw}


@router.post("/auth/login")
def login(body: Credentials, session: Session = Depends(get_session)):
    try:
        user = service.authenticate(session, body.email, body.password)
    except AccountError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    _key, raw = service.create_api_key(session, user, name="web-session", kind="session")
    return {"user": _user_public(user), "token": raw}


@router.get("/auth/me")
def me(user: User = Depends(require_user)):
    return _user_public(user)


@router.get("/keys")
def list_keys(user: User = Depends(require_user), session: Session = Depends(get_session)):
    return [
        {"id": k.id, "name": k.name, "prefix": k.prefix, "kind": k.kind,
         "created_at": k.created_at.isoformat(), "last_used": k.last_used.isoformat() if k.last_used else None}
        for k in service.list_keys(session, user)
    ]


@router.post("/keys")
def create_key(body: KeyRequest, user: User = Depends(require_user), session: Session = Depends(get_session)):
    _key, raw = service.create_api_key(session, user, name=body.name)
    return {"api_key": raw, "name": body.name}


@router.delete("/keys/{key_id}")
def revoke_key(key_id: str, user: User = Depends(require_user), session: Session = Depends(get_session)):
    if not service.revoke_key(session, user, key_id):
        raise HTTPException(status_code=404, detail="key not found")
    return {"revoked": key_id}
