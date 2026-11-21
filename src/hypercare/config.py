"""Typed settings: defaults < .env file < HYPERCARE_* environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_DIR = Path(__file__).parent
DATA_DIR = PACKAGE_DIR / "data"

Mode = Literal["live", "replay"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HYPERCARE_", env_file=".env", extra="ignore")

    # Storage
    db_path: Path = Path("var/hypercare.db")
    seed_dir: Path = DATA_DIR / "seed"
    kb_dir: Path = DATA_DIR / "kb"
    replay_path: Path = DATA_DIR / "replays.json"

    # LLM
    mode: Mode = "replay"
    model: str = "claude-opus-5-5"
    effort: Effort = "medium"
    max_tokens: int = 16000
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    # Agents
    retrieval: Literal["bm25", "hybrid"] = "bm25"
    """``hybrid`` adds dense embeddings (needs the ``embeddings`` extra)."""
    retrieval_top_k: int = 4
    recurring_min_tickets: int = 4
    """A subcategory with at least this many open tickets is a recurring pattern (KB gap)."""


@lru_cache
def get_settings() -> Settings:
    return Settings()
