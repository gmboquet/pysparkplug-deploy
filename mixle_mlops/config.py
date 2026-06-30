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
    default_model: str = "echo"

    # --- cloud backends (deployment == "cloud") ---
    s3_bucket: str | None = None
    s3_endpoint: str | None = None

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
