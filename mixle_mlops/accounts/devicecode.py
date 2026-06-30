"""RFC 8628 device-authorization grant.

Lets a device with no browser (the CLI agent) sign in: it requests a device code, shows the user a short
``user_code`` + a verification URL, the user approves in any browser (password / Google / Apple), and the
device polls until it receives a mixle API token.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, select

from ..config import Settings, get_settings
from . import service
from .models import DeviceCode, User

# Crockford-ish alphabet: no 0/O/1/I to keep user codes unambiguous when typed.
_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime) -> datetime:
    # SQLite returns tz-naive datetimes; treat stored times as UTC for comparison.
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _user_code() -> str:
    raw = "".join(secrets.choice(_ALPHABET) for _ in range(8))
    return f"{raw[:4]}-{raw[4:]}"


def _hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def new_device_code(session: Session, settings: Settings | None = None) -> dict:
    s = settings or get_settings()
    raw = secrets.token_urlsafe(32)
    user_code = _user_code()
    rec = DeviceCode(
        device_code_hash=_hash(raw),
        user_code=user_code,
        expires_at=_now() + timedelta(seconds=s.oauth_device_ttl),
        interval=5,
    )
    session.add(rec)
    session.commit()
    verification_uri = f"{s.public_url}/auth/device"
    return {
        "device_code": raw,
        "user_code": user_code,
        "verification_uri": verification_uri,
        "verification_uri_complete": f"{verification_uri}?user_code={user_code}",
        "expires_in": s.oauth_device_ttl,
        "interval": rec.interval,
    }


def _by_user_code(session: Session, user_code: str) -> DeviceCode | None:
    return session.exec(
        select(DeviceCode).where(DeviceCode.user_code == user_code.strip().upper())
    ).first()


def approve(session: Session, user_code: str, user: User) -> bool:
    rec = _by_user_code(session, user_code)
    if rec is None or rec.status != "pending" or _aware(rec.expires_at) < _now():
        return False
    rec.status = "approved"
    rec.user_id = user.id
    session.add(rec)
    session.commit()
    return True


def deny(session: Session, user_code: str, user: User) -> bool:
    rec = _by_user_code(session, user_code)
    if rec is None or rec.status != "pending":
        return False
    rec.status = "denied"
    session.add(rec)
    session.commit()
    return True


def poll(session: Session, device_code: str) -> dict:
    """Returns {status: pending|denied|expired} or {status: approved, token, user}."""
    rec = session.exec(
        select(DeviceCode).where(DeviceCode.device_code_hash == _hash(device_code))
    ).first()
    if rec is None:
        return {"status": "invalid"}
    if rec.status in ("denied", "expired"):
        return {"status": rec.status}
    if _aware(rec.expires_at) < _now():
        rec.status = "expired"
        session.add(rec)
        session.commit()
        return {"status": "expired"}
    if rec.status == "claimed":
        return {"status": "expired"}  # one-shot: a device code yields a token exactly once
    if rec.status == "approved" and rec.user_id:
        user = session.get(User, rec.user_id)
        if user is None:
            return {"status": "denied"}
        _key, raw = service.create_api_key(session, user, name="cli-device", kind="device")
        rec.status = "claimed"
        session.add(rec)
        session.commit()
        return {"status": "approved", "token": raw, "user": user}
    return {"status": "pending"}
