"""OIDC sign-in (Sign in with Google / Apple).

The gateway is the OAuth *client* and identity broker: it sends the user to the provider, exchanges the
authorization code for an ``id_token``, verifies it against the provider's JWKS, and find-or-creates a
mixle ``User`` linked to that external identity. State is a stateless signed token (HMAC with
``secret_key``) so no server-side session store is needed for the redirect round-trip.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from urllib.parse import urlencode, urlparse

import httpx
import jwt
from sqlmodel import Session, select

from ..config import Settings, get_settings
from .models import OAuthIdentity, User


class OAuthError(Exception):
    pass


@dataclass
class OAuthProvider:
    name: str
    client_id: str
    client_secret: str
    issuer: str
    jwks_uri: str
    auth_uri: str
    token_uri: str
    scope: str


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64u_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def apple_client_secret(settings: Settings) -> str:
    """Apple's client secret is an ES256 JWT signed with the .p8 key (valid up to 6 months)."""
    now = int(time.time())
    return jwt.encode(
        {
            "iss": settings.apple_team_id,
            "iat": now,
            "exp": now + 3600,
            "aud": settings.apple_issuer,
            "sub": settings.apple_client_id,
        },
        settings.apple_private_key,
        algorithm="ES256",
        headers={"kid": settings.apple_key_id},
    )


def get_provider(name: str, settings: Settings | None = None) -> OAuthProvider | None:
    """Return a configured provider, or None if its client_id is unset (provider disabled)."""
    s = settings or get_settings()
    if name == "google" and s.google_client_id:
        return OAuthProvider(
            name="google",
            client_id=s.google_client_id,
            client_secret=s.google_client_secret,
            issuer=s.google_issuer,
            jwks_uri=s.google_jwks_uri,
            auth_uri=s.google_auth_uri,
            token_uri=s.google_token_uri,
            scope="openid email profile",
        )
    if name == "apple" and s.apple_client_id:
        return OAuthProvider(
            name="apple",
            client_id=s.apple_client_id,
            client_secret=apple_client_secret(s) if s.apple_private_key else "",
            issuer=s.apple_issuer,
            jwks_uri=s.apple_jwks_uri,
            auth_uri=s.apple_auth_uri,
            token_uri=s.apple_token_uri,
            scope="openid email name",
        )
    return None


def enabled_providers(settings: Settings | None = None) -> list[str]:
    return [name for name in ("google", "apple") if get_provider(name, settings) is not None]


# --- stateless signed state (binds nonce + redirect_uri across the redirect round-trip) ---


def make_state(
    nonce: str, redirect_uri: str, redirect_to: str | None = None, settings: Settings | None = None
) -> str:
    s = settings or get_settings()
    payload = {
        "nonce": nonce,
        "redirect_uri": redirect_uri,
        "redirect_to": redirect_to,
        "exp": int(time.time()) + s.oauth_state_ttl,
    }
    body = _b64u(json.dumps(payload, separators=(",", ":")).encode())
    sig = _b64u(hmac.new(s.secret_key.encode(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def read_state(state: str, settings: Settings | None = None) -> dict:
    s = settings or get_settings()
    try:
        body, sig = state.split(".", 1)
    except ValueError as exc:
        raise OAuthError("malformed state") from exc
    expect = _b64u(hmac.new(s.secret_key.encode(), body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expect):
        raise OAuthError("invalid state signature")
    payload = json.loads(_b64u_decode(body))
    if payload.get("exp", 0) < int(time.time()):
        raise OAuthError("state expired")
    return payload


def is_allowed_redirect(redirect_to: str | None, settings: Settings | None = None) -> bool:
    """Guard against open-redirect / token leak: only same-origin destinations are allowed."""
    if not redirect_to:
        return True  # no post-login redirect -> JSON response (no fragment token)
    if redirect_to.startswith("/") and not redirect_to.startswith("//"):
        return True  # same-origin relative path
    s = settings or get_settings()
    try:
        u = urlparse(redirect_to)
    except ValueError:
        return False
    if u.scheme not in ("http", "https"):
        return False
    target = (u.scheme, u.hostname, u.port)
    allowed = [s.public_url]
    if s.oauth_web_origin:
        allowed.append(s.oauth_web_origin)
    for origin in allowed:
        o = urlparse(origin)
        if target == (o.scheme, o.hostname, o.port):
            return True
    return False


def authorization_url(
    provider: OAuthProvider,
    redirect_uri: str,
    redirect_to: str | None = None,
    settings: Settings | None = None,
) -> dict:
    nonce = secrets.token_urlsafe(24)  # cryptographically-random (was sha256 of wall-clock time)
    state = make_state(nonce, redirect_uri, redirect_to, settings)
    params = {
        "client_id": provider.client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": provider.scope,
        "state": state,
        "nonce": nonce,
    }
    if provider.name == "apple":
        params["response_mode"] = "form_post"
    return {"url": f"{provider.auth_uri}?{urlencode(params)}", "state": state, "nonce": nonce}


def exchange_code(provider: OAuthProvider, code: str, redirect_uri: str) -> dict:
    """Exchange the authorization code for tokens at the provider's token endpoint."""
    resp = httpx.post(
        provider.token_uri,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": provider.client_id,
            "client_secret": provider.client_secret,
        },
        headers={"Accept": "application/json"},
        timeout=15.0,
    )
    if resp.status_code != 200:
        raise OAuthError(f"token exchange failed ({resp.status_code}): {resp.text[:200]}")
    return resp.json()


# Indirection so tests can supply a signing key without reaching the network.
def _signing_key(provider: OAuthProvider, token: str):
    return jwt.PyJWKClient(provider.jwks_uri).get_signing_key_from_jwt(token).key


def verify_id_token(provider: OAuthProvider, id_token: str, nonce: str | None = None) -> dict:
    """Verify signature, audience, issuer, expiry, and (optionally) nonce; return the claims."""
    try:
        key = _signing_key(provider, id_token)
        claims = jwt.decode(
            id_token,
            key,
            algorithms=["RS256", "ES256"],
            audience=provider.client_id,
            issuer=provider.issuer,
        )
    except jwt.PyJWTError as exc:
        raise OAuthError(f"invalid id_token: {exc}") from exc
    if nonce is not None and claims.get("nonce") != nonce:
        raise OAuthError("nonce mismatch")
    if not claims.get("sub"):
        raise OAuthError("id_token missing subject")
    return claims


def find_or_create_user(session: Session, provider_name: str, claims: dict) -> User:
    sub = claims["sub"]
    email = claims.get("email")
    # Only trust the email for cross-account linking if the provider says it is verified —
    # otherwise an attacker could register an OIDC identity with a victim's email and take over
    # the victim's existing password account.
    verified = claims.get("email_verified") in (True, "true", "True", 1)
    ident = session.exec(
        select(OAuthIdentity).where(
            OAuthIdentity.provider == provider_name, OAuthIdentity.subject == sub
        )
    ).first()
    if ident is not None:
        user = session.get(User, ident.user_id)
        if user is not None:
            return user
    user = None
    if email and verified:
        user = session.exec(select(User).where(User.email == email)).first()
    if user is None:
        # Verified email becomes the account email; otherwise key by (provider, sub) so we never
        # collide with — or hijack — an existing account.
        account_email = email if (email and verified) else f"{provider_name}_{sub}@users.mixle.local"
        user = User(email=account_email)
        session.add(user)
        session.commit()
        session.refresh(user)
    session.add(
        OAuthIdentity(user_id=user.id, provider=provider_name, subject=sub, email=email)
    )
    session.commit()
    return user
