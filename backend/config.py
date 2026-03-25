"""
Configuration settings for the Music Generator backend.
Loaded from environment variables with sensible defaults.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    # Default model
    default_model: str = "facebook/musicgen-melody"

    # Whisper
    whisper_model_size: str = "base"   # tiny | base | small | medium | large

    # Max upload size (bytes) — 50 MB
    max_upload_bytes: int = 50 * 1024 * 1024

    # Output directory (relative to backend/)
    output_dir: str = "outputs"

    class Config:
        env_prefix = "MUSIC_GEN_"
        env_file = ".env"


settings = Settings()
