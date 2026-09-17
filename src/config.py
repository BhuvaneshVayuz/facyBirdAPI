"""Central configuration. Everything tunable lives here or in .env."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    cors_allow_origins: str = "*"
    #: Rejects abusive/oversized downloads before they ever reach decode/rembg.
    #: Same limit whether the bytes came from a multipart upload or (now) a
    #: server-side download of a host-supplied photo_url.
    max_upload_bytes: int = 8 * 1024 * 1024
    #: Final cutout is square; this is one side, in pixels.
    output_size: int = 512
    #: How long the server will wait on a host-supplied photo_url before
    #: giving up -- this request blocks the caller's whole Facy Bird flow,
    #: so a slow/hanging photo host shouldn't be able to stall it forever.
    fetch_timeout_seconds: float = 10.0

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


settings = Settings()
