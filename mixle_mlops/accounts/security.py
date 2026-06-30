"""Password hashing (PBKDF2 + a config pepper) and API-key generation/hashing — stdlib only, no native deps."""
from __future__ import annotations

import hashlib
import hmac
import secrets

from ..config import get_settings

_ITERATIONS = 200_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    pepper = get_settings().secret_key
    dk = hashlib.pbkdf2_hmac("sha256", (password + pepper).encode(), bytes.fromhex(salt), _ITERATIONS)
    return f"pbkdf2${_ITERATIONS}${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _algo, iters, salt, hexhash = stored.split("$")
        pepper = get_settings().secret_key
        dk = hashlib.pbkdf2_hmac("sha256", (password + pepper).encode(), bytes.fromhex(salt), int(iters))
        return hmac.compare_digest(dk.hex(), hexhash)
    except (ValueError, AttributeError):
        return False


def generate_api_key() -> tuple[str, str, str]:
    """Return (raw_key, prefix, sha256_hash). The raw key is shown to the user once and never stored."""
    raw = "mk-" + secrets.token_urlsafe(32)
    return raw, raw[:12], hashlib.sha256(raw.encode()).hexdigest()


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()
