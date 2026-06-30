"""Identity routes: signup, login, and API-key management (used by the chat UI and by API clients)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlmodel import Session

from ...accounts import devicecode, oauth, service
from ...accounts.models import User
from ...accounts.oauth import OAuthError
from ...accounts.service import AccountError
from ...config import get_settings
from ...storage.db import get_session
from ..auth import require_user

router = APIRouter()


class Credentials(BaseModel):
    email: str
    password: str


class KeyRequest(BaseModel):
    name: str = "default"


class DeviceTokenRequest(BaseModel):
    device_code: str


class UserCodeRequest(BaseModel):
    user_code: str


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


# --------------------------------------------------------------------------- #
# OAuth / OIDC sign-in (Sign in with Google / Apple)
# --------------------------------------------------------------------------- #


@router.get("/auth/providers")
def providers():
    """Which sign-in methods this deployment offers (password is always available)."""
    return {"password": True, "oauth": oauth.enabled_providers()}


def _default_redirect(provider: str) -> str:
    return f"{get_settings().public_url}/auth/oauth/{provider}/callback"


@router.get("/auth/oauth/{provider}/url")
def oauth_url(provider: str, redirect_uri: str | None = None):
    prov = oauth.get_provider(provider)
    if prov is None:
        raise HTTPException(status_code=404, detail=f"provider '{provider}' is not enabled")
    return oauth.authorization_url(prov, redirect_uri or _default_redirect(provider))


def _complete_login(provider: str, code: str, state: str, session: Session) -> dict:
    prov = oauth.get_provider(provider)
    if prov is None:
        raise HTTPException(status_code=404, detail=f"provider '{provider}' is not enabled")
    try:
        payload = oauth.read_state(state)
        tokens = oauth.exchange_code(prov, code, payload["redirect_uri"])
        id_token = tokens.get("id_token")
        if not id_token:
            raise OAuthError("provider did not return an id_token")
        claims = oauth.verify_id_token(prov, id_token, nonce=payload.get("nonce"))
    except OAuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    user = oauth.find_or_create_user(session, provider, claims)
    _key, raw = service.create_api_key(session, user, name=f"{provider}-session", kind="session")
    return {"user": _user_public(user), "token": raw}


@router.get("/auth/oauth/{provider}/callback")
def oauth_callback_get(
    provider: str, code: str, state: str, session: Session = Depends(get_session)
):
    return _complete_login(provider, code, state, session)


@router.post("/auth/oauth/{provider}/callback")
async def oauth_callback_post(
    provider: str, request: Request, session: Session = Depends(get_session)
):
    # Apple uses response_mode=form_post.
    form = await request.form()
    code = form.get("code")
    state = form.get("state")
    if not code or not state:
        raise HTTPException(status_code=400, detail="missing code or state")
    return _complete_login(provider, str(code), str(state), session)


# --------------------------------------------------------------------------- #
# Device-authorization grant (CLI agent login)
# --------------------------------------------------------------------------- #


@router.post("/auth/device/code")
def device_code(session: Session = Depends(get_session)):
    return devicecode.new_device_code(session)


@router.post("/auth/device/token")
def device_token(body: DeviceTokenRequest, session: Session = Depends(get_session)):
    result = devicecode.poll(session, body.device_code)
    status = result["status"]
    if status == "approved":
        return {"token": result["token"], "user": _user_public(result["user"])}
    # OAuth device-grant error semantics
    mapping = {
        "pending": "authorization_pending",
        "denied": "access_denied",
        "expired": "expired_token",
        "invalid": "invalid_grant",
    }
    raise HTTPException(status_code=400, detail={"error": mapping.get(status, "invalid_grant")})


@router.post("/auth/device/approve")
def device_approve(
    body: UserCodeRequest, user: User = Depends(require_user), session: Session = Depends(get_session)
):
    if not devicecode.approve(session, body.user_code, user):
        raise HTTPException(status_code=404, detail="unknown or already-resolved code")
    return {"approved": body.user_code}


@router.post("/auth/device/deny")
def device_deny(
    body: UserCodeRequest, user: User = Depends(require_user), session: Session = Depends(get_session)
):
    if not devicecode.deny(session, body.user_code, user):
        raise HTTPException(status_code=404, detail="unknown or already-resolved code")
    return {"denied": body.user_code}
