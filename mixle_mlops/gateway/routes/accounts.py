"""Identity routes: signup, login, and API-key management (used by the chat UI and by API clients)."""
from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
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
def oauth_url(provider: str, redirect_uri: str | None = None, redirect_to: str | None = None):
    prov = oauth.get_provider(provider)
    if prov is None:
        raise HTTPException(status_code=404, detail=f"provider '{provider}' is not enabled")
    if not oauth.is_allowed_redirect(redirect_to):
        raise HTTPException(status_code=400, detail="redirect_to is not an allowed (same-origin) destination")
    return oauth.authorization_url(prov, redirect_uri or _default_redirect(provider), redirect_to)


def _complete_login(provider: str, code: str, state: str, session: Session):
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
    redirect_to = payload.get("redirect_to")
    if redirect_to and not oauth.is_allowed_redirect(redirect_to):
        redirect_to = None  # defense-in-depth: never redirect a token to a non-same-origin URL
    if redirect_to:
        # Browser flow: hand the token back to the page via the URL fragment (never sent to a server).
        sep = "&" if "#" in redirect_to else "#"
        return RedirectResponse(
            f"{redirect_to}{sep}token={quote(raw)}&email={quote(user.email)}", status_code=303
        )
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


@router.get("/auth/device", response_class=HTMLResponse)
def device_page():
    """Self-contained approval page the CLI directs users to (password / Google / Apple)."""
    return HTMLResponse(_DEVICE_PAGE)


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


_DEVICE_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Approve device · mixle</title>
<style>
 body{font-family:ui-sans-serif,system-ui;background:#0f1117;color:#e6e8ee;display:flex;
   min-height:100vh;align-items:center;justify-content:center;margin:0}
 .card{background:#171a23;border:1px solid #262b38;border-radius:14px;padding:28px;width:340px}
 h1{font-size:18px;margin:0 0 4px} .sub{color:#8a90a2;font-size:13px;margin-bottom:18px}
 input{width:100%;box-sizing:border-box;padding:10px;margin:6px 0;border-radius:8px;
   border:1px solid #262b38;background:#0f1117;color:#e6e8ee}
 button{width:100%;padding:10px;border:none;border-radius:8px;background:#b06bcf;color:#fff;
   cursor:pointer;font-weight:600;margin-top:6px}
 .oauth button{background:#222735;border:1px solid #2e3447}
 .code{font-family:ui-monospace,monospace;font-size:20px;letter-spacing:2px;text-align:center;
   background:#0f1117;border:1px dashed #3a4150;border-radius:8px;padding:10px;margin:10px 0}
 .msg{margin-top:14px;font-size:13px} .ok{color:#3fb950} .err{color:#f85149}
 .hr{height:1px;background:#262b38;margin:16px 0}
</style></head>
<body><div class="card">
 <h1>Authorize mixle-agent</h1>
 <div class="sub">Sign in to approve this device.</div>
 <div>Device code</div>
 <div class="code" id="code">----</div>
 <div id="auth">
   <input id="email" type="email" placeholder="email" autocomplete="username"/>
   <input id="password" type="password" placeholder="password" autocomplete="current-password"/>
   <button id="primary" onclick="submitAuth()">Sign in &amp; approve</button>
   <div style="margin-top:10px;font-size:13px;text-align:center">
     <a href="#" id="toggle" onclick="toggleMode(event)">Create an account</a>
   </div>
   <div class="oauth" id="oauth"></div>
 </div>
 <div class="msg" id="msg"></div>
</div>
<script>
const qs = new URLSearchParams(location.search);
const userCode = (qs.get("user_code")||"").toUpperCase();
document.getElementById("code").textContent = userCode || "(missing)";
const msg = (t, ok) => { const m=document.getElementById("msg"); m.textContent=t; m.className="msg "+(ok?"ok":"err"); };

async function approve(token){
  const r = await fetch("/auth/device/approve", {method:"POST",
    headers:{"content-type":"application/json","Authorization":"Bearer "+token},
    body: JSON.stringify({user_code:userCode})});
  if(r.ok){ document.getElementById("auth").style.display="none";
    msg("✓ Approved. You can return to your terminal.", true); }
  else { const e=await r.json().catch(()=>({})); msg("Could not approve: "+(e.detail||r.status), false); }
}
let mode = "login";
function toggleMode(e){
  e.preventDefault();
  mode = mode === "login" ? "signup" : "login";
  document.getElementById("primary").textContent =
    mode === "login" ? "Sign in & approve" : "Create account & approve";
  document.getElementById("toggle").textContent =
    mode === "login" ? "Create an account" : "I already have an account";
  msg("", true);
}
async function submitAuth(){
  const email=document.getElementById("email").value.trim(), password=document.getElementById("password").value;
  if(!email || !password){ msg("Enter an email and password.", false); return; }
  const path = mode === "signup" ? "/auth/signup" : "/auth/login";
  const r = await fetch(path,{method:"POST",headers:{"content-type":"application/json"},
    body:JSON.stringify({email,password})});
  if(!r.ok){
    const e = await r.json().catch(()=>({}));
    msg(mode === "signup" ? ("Could not create account: " + (e.detail||r.status)) : "Invalid email or password.", false);
    return;
  }
  const d = await r.json(); await approve(d.token || d.api_key);
}
async function loadProviders(){
  try{ const d = await (await fetch("/auth/providers")).json();
    const box=document.getElementById("oauth");
    if((d.oauth||[]).length){ box.innerHTML='<div class="hr"></div>'; }
    for(const p of (d.oauth||[])){
      const b=document.createElement("button"); b.textContent="Sign in with "+p[0].toUpperCase()+p.slice(1);
      b.onclick=async()=>{ const redirect="/auth/device?user_code="+encodeURIComponent(userCode);
        const u=await (await fetch("/auth/oauth/"+p+"/url?redirect_to="+encodeURIComponent(redirect))).json();
        location.href=u.url; };
      box.appendChild(b);
    }
  }catch(e){}
}
// returning from an OAuth round-trip: token arrives in the URL fragment
const hash=new URLSearchParams(location.hash.slice(1));
if(hash.get("token")){ approve(hash.get("token")); history.replaceState(null,"",location.pathname+location.search); }
loadProviders();
</script>
</body></html>"""
