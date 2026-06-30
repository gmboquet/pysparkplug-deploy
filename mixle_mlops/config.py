"""Platform configuration (env-driven, prefix ``MIXLE_``). Local-first: the same settings scale to a cloud
deployment by changing values (sqlite→postgres, filesystem→s3), no code change."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MIXLE_", env_file=".env", extra="ignore", protected_namespaces=()
    )

    # --- deployment ---
    deployment: str = "local"                       # "local" (sqlite + fs) or "cloud" (postgres + s3)
    data_dir: Path = Path("./mixle_data")
    database_url: str | None = None                 # overrides the deployment default
    secret_key: str = "dev-insecure-change-me"      # pepper for password/token hashing
    require_auth: bool = True                        # False = allow anonymous access (local dev only)
    cors_origins: list[str] = ["*"]

    # --- model registry ---
    registry_root: Path = Path("./mixle_data/registry")
    enable_demo_models: bool = True                  # register a small fitted mixle model to demo the /v1/mixle routes

    # --- default LLM backend: any OpenAI-compatible server (Ollama :11434/v1, vLLM, llama.cpp, hosted) ---
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_models: list[str] = []                       # ids to expose from the LLM backend ([] = discover)
    # per-model backends — host local + cloud models in ONE registry (the cascade router prerequisite):
    #   MIXLE_LLM_BACKENDS='{"llama3.2":{"base_url":"http://ollama:11434/v1"},
    #                        "frontier":{"base_url":"https://api.openai.com/v1","api_key":"sk-...","upstream_model":"gpt-4o"}}'
    llm_backends: dict[str, dict[str, str]] = {}
    default_model: str = "echo"

    # --- local logit-level engine (token-level PoE + grammar masking via transformers; needs the `local` extra) ---
    local_model: str = ""                            # a transformers model id to host through the decode engine
    local_poe_models: list[str] = []                 # 2+ model ids -> a token-level Product-of-Experts ensemble
    local_max_tokens: int = 128

    # --- self-evolution ---
    # >0 runs an autonomous improve pass over all hosted mixle models every N seconds. Off by default; run it on
    # ONE instance only (not every replica) — it mutates shared served models. Anti-regression: a pass can only
    # improve or no-op a model, never degrade it.
    evolve_interval_seconds: int = 0

    # --- scale / cache / concurrency ---
    redis_url: str | None = None                     # MIXLE_REDIS_URL: shared cache + rate-limit across replicas
    enable_response_cache: bool = False              # cache (exact + semantic) chat completions
    rate_limit_per_min: int = 0                      # 0 = disabled; else max requests/min per api key

    # --- image-generation backend (OpenAI-compatible /v1/images/generations) ---
    image_base_url: str = ""
    image_api_key: str = ""
    image_model: str = ""

    # --- cloud backends (deployment == "cloud") ---
    s3_bucket: str | None = None
    s3_endpoint: str | None = None

    # --- OAuth / OIDC sign-in ("Sign in with Google / Apple") ---
    public_url: str = "http://localhost:8000"        # base URL of this gateway (OAuth redirect + device verification)
    oauth_device_ttl: int = 600                      # seconds a device code stays valid
    oauth_state_ttl: int = 600                       # seconds an OAuth state token stays valid
    # Google
    google_client_id: str = ""
    google_client_secret: str = ""
    google_issuer: str = "https://accounts.google.com"
    google_jwks_uri: str = "https://www.googleapis.com/oauth2/v3/certs"
    google_auth_uri: str = "https://accounts.google.com/o/oauth2/v2/auth"
    google_token_uri: str = "https://oauth2.googleapis.com/token"
    # Apple (Sign in with Apple). The "client secret" is an ES256 JWT signed with the .p8 key.
    apple_client_id: str = ""                         # the Services ID
    apple_team_id: str = ""
    apple_key_id: str = ""
    apple_private_key: str = ""                       # contents of the .p8 private key (PEM)
    apple_issuer: str = "https://appleid.apple.com"
    apple_jwks_uri: str = "https://appleid.apple.com/auth/keys"
    apple_auth_uri: str = "https://appleid.apple.com/auth/authorize"
    apple_token_uri: str = "https://appleid.apple.com/auth/token"

    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        if self.deployment == "cloud":
            raise RuntimeError("cloud deployment requires MIXLE_DATABASE_URL (e.g. postgresql+psycopg://...)")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{(self.data_dir / 'platform.db').resolve()}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
