"""LLM clients: live (Anthropic), replay (recorded outputs) and a fake for tests."""

from __future__ import annotations

from hypercare.config import Settings
from hypercare.llm.base import (
    LLMClient,
    LLMError,
    LLMResult,
    LLMUnavailableError,
    Usage,
    estimate_cost,
    prompt_key,
)
from hypercare.llm.replay import RecordingClient, ReplayClient

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMResult",
    "LLMUnavailableError",
    "RecordingClient",
    "ReplayClient",
    "Usage",
    "build_llm",
    "estimate_cost",
    "prompt_key",
]


def build_llm(settings: Settings, *, record: bool = False) -> LLMClient:
    """Return the client for the configured mode. ``record`` saves live calls for replay."""
    if settings.mode == "replay":
        return ReplayClient(settings.replay_path)

    from hypercare.llm.anthropic_client import AnthropicClient  # noqa: PLC0415 - SDK only in live mode

    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
    if not key:
        raise LLMError("live mode needs ANTHROPIC_API_KEY (in the environment or in .env)")
    live = AnthropicClient(
        model=settings.model, effort=settings.effort, max_tokens=settings.max_tokens, api_key=key
    )
    return RecordingClient(live, settings.replay_path) if record else live
