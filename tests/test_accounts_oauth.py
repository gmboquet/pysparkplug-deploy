"""Device-authorization grant + OIDC sign-in (Google/Apple). OIDC is exercised offline with a locally
generated RSA key standing in for the provider's JWKS, so no network or real client IDs are needed."""
import jwt
import mixle_mlops.storage.db as db
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from mixle_mlops.accounts import oauth
from mixle_mlops.config import get_settings
from mixle_mlops.gateway.app import create_app


def _make_client(tmp_path, monkeypatch, **env):
    monkeypatch.setenv("MIXLE_DATA_DIR", str(tmp_path))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    db._engine = None
    return TestClient(create_app())


@pytest.fixture
def client(tmp_path, monkeypatch):
    with _make_client(tmp_path, monkeypatch) as c:
        yield c
    get_settings.cache_clear()
    db._engine = None


def _signup(client, email="dev@mixle.local"):
    return client.post("/auth/signup", json={"email": email, "password": "pw123456"}).json()["api_key"]


def test_device_flow_end_to_end(client):
    # 1. CLI requests a device code
    dc = client.post("/auth/device/code").json()
    assert dc["user_code"] and dc["device_code"] and dc["verification_uri"]

    # 2. polling before approval -> authorization_pending
    pending = client.post("/auth/device/token", json={"device_code": dc["device_code"]})
    assert pending.status_code == 400
    assert pending.json()["detail"]["error"] == "authorization_pending"

    # 3. a logged-in user approves the code in the browser
    token = _signup(client)
    headers = {"Authorization": f"Bearer {token}"}
    approved = client.post("/auth/device/approve", json={"user_code": dc["user_code"]}, headers=headers)
    assert approved.status_code == 200

    # 4. the CLI polls again and now receives a token that actually works
    granted = client.post("/auth/device/token", json={"device_code": dc["device_code"]}).json()
    assert granted["token"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {granted['token']}"})
    assert me.status_code == 200 and me.json()["email"] == "dev@mixle.local"

    # 5. a device code is one-shot
    again = client.post("/auth/device/token", json={"device_code": dc["device_code"]})
    assert again.status_code == 400


def test_device_deny(client):
    dc = client.post("/auth/device/code").json()
    token = _signup(client, "deny@mixle.local")
    client.post("/auth/device/deny", json={"user_code": dc["user_code"]},
                headers={"Authorization": f"Bearer {token}"})
    r = client.post("/auth/device/token", json={"device_code": dc["device_code"]})
    assert r.status_code == 400 and r.json()["detail"]["error"] == "access_denied"


def test_providers_listing(tmp_path, monkeypatch):
    with _make_client(tmp_path, monkeypatch) as c:
        assert c.get("/auth/providers").json() == {"password": True, "oauth": []}
    get_settings.cache_clear()
    db._engine = None
    with _make_client(tmp_path, monkeypatch, MIXLE_GOOGLE_CLIENT_ID="gid.apps.googleusercontent.com") as c:
        assert "google" in c.get("/auth/providers").json()["oauth"]
    get_settings.cache_clear()
    db._engine = None


def test_google_oidc_signin(tmp_path, monkeypatch):
    client_id = "gid.apps.googleusercontent.com"
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    pub_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()

    with _make_client(tmp_path, monkeypatch, MIXLE_GOOGLE_CLIENT_ID=client_id) as c:
        # the gateway hands us an authorization URL with a signed state + nonce
        started = c.get("/auth/oauth/google/url").json()
        assert client_id in started["url"]
        nonce = started["nonce"]
        state = started["state"]

        id_token = jwt.encode(
            {
                "iss": "https://accounts.google.com",
                "aud": client_id,
                "sub": "google-user-123",
                "email": "alice@gmail.com",
                "email_verified": True,
                "nonce": nonce,
                "exp": 9999999999,
            },
            priv_pem,
            algorithm="RS256",
        )
        # stub the network bits: code-exchange + JWKS signing key
        monkeypatch.setattr(oauth, "exchange_code", lambda prov, code, redirect_uri: {"id_token": id_token})
        monkeypatch.setattr(oauth, "_signing_key", lambda prov, token: pub_pem)

        res = c.get("/auth/oauth/google/callback", params={"code": "authz-code", "state": state})
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["user"]["email"] == "alice@gmail.com" and body["token"]

        me = c.get("/auth/me", headers={"Authorization": f"Bearer {body['token']}"})
        assert me.status_code == 200 and me.json()["email"] == "alice@gmail.com"

        # signing in again with the same provider subject reuses the same account
        res2 = c.get("/auth/oauth/google/callback", params={"code": "authz-code-2", "state": state})
        assert res2.json()["user"]["id"] == body["user"]["id"]

    get_settings.cache_clear()
    db._engine = None


def test_oauth_unknown_provider(client):
    assert client.get("/auth/oauth/google/url").status_code == 404   # not enabled by default


def test_device_page_renders(client):
    r = client.get("/auth/device", params={"user_code": "ABCD-EFGH"})
    assert r.status_code == 200
    assert "Authorize mixle-agent" in r.text
    assert "text/html" in r.headers["content-type"]


def test_oauth_browser_redirect_flow(tmp_path, monkeypatch):
    client_id = "gid.apps.googleusercontent.com"
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    pub_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    with _make_client(tmp_path, monkeypatch, MIXLE_GOOGLE_CLIENT_ID=client_id) as c:
        started = c.get("/auth/oauth/google/url", params={"redirect_to": "http://localhost:8000/auth/device?user_code=ABCD-EFGH"}).json()
        id_token = jwt.encode(
            {"iss": "https://accounts.google.com", "aud": client_id, "sub": "u1",
             "email": "bob@gmail.com", "email_verified": True, "nonce": started["nonce"], "exp": 9999999999},
            priv_pem, algorithm="RS256",
        )
        monkeypatch.setattr(oauth, "exchange_code", lambda p, code, ru: {"id_token": id_token})
        monkeypatch.setattr(oauth, "_signing_key", lambda p, t: pub_pem)
        res = c.get("/auth/oauth/google/callback",
                    params={"code": "x", "state": started["state"]}, follow_redirects=False)
        assert res.status_code == 303
        loc = res.headers["location"]
        assert loc.startswith("http://localhost:8000/auth/device?user_code=ABCD-EFGH#")
        assert "token=" in loc and "email=" in loc
    get_settings.cache_clear()
    db._engine = None


def test_oauth_rejects_open_redirect(tmp_path, monkeypatch):
    with _make_client(tmp_path, monkeypatch, MIXLE_GOOGLE_CLIENT_ID="gid") as c:
        # off-origin destinations are rejected (would otherwise leak the token via the fragment)
        for evil in ["https://evil.com", "//evil.com", "http://localhost:9999/x"]:
            assert c.get("/auth/oauth/google/url", params={"redirect_to": evil}).status_code == 400
        # same-origin relative + absolute are allowed
        assert c.get("/auth/oauth/google/url", params={"redirect_to": "/auth/device"}).status_code == 200
        assert c.get("/auth/oauth/google/url",
                     params={"redirect_to": "http://localhost:8000/auth/device"}).status_code == 200
    get_settings.cache_clear()
    db._engine = None


def test_unverified_oidc_email_does_not_hijack_account(tmp_path, monkeypatch):
    client_id = "gid.apps.googleusercontent.com"
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    pub_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    with _make_client(tmp_path, monkeypatch, MIXLE_GOOGLE_CLIENT_ID=client_id) as c:
        # an existing password account
        victim = c.post("/auth/signup", json={"email": "victim@corp.com", "password": "pw123456"}).json()
        started = c.get("/auth/oauth/google/url").json()
        # attacker's OIDC identity claims the victim's email but it is NOT verified
        id_token = jwt.encode(
            {"iss": "https://accounts.google.com", "aud": client_id, "sub": "attacker-sub",
             "email": "victim@corp.com", "email_verified": False, "nonce": started["nonce"], "exp": 9999999999},
            priv_pem, algorithm="RS256",
        )
        monkeypatch.setattr(oauth, "exchange_code", lambda p, code, ru: {"id_token": id_token})
        monkeypatch.setattr(oauth, "_signing_key", lambda p, t: pub_pem)
        body = c.get("/auth/oauth/google/callback", params={"code": "x", "state": started["state"]}).json()
        # must NOT resolve to the victim's account, and must not claim their email
        assert body["user"]["id"] != victim["user"]["id"]
        assert body["user"]["email"] != "victim@corp.com"
    get_settings.cache_clear()
    db._engine = None
    get_settings.cache_clear()
    db._engine = None
