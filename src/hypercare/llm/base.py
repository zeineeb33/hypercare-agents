"""LLM abstraction: agents ask for a structured output and get a validated Pydantic object back."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel


class LLMError(Exception):
    """The model could not produce a usable answer (API error, refusal, truncated or invalid output).

    Agents treat this as "no evidence" and the pipeline fails closed (escalation).
    """


class LLMUnavailableError(LLMError):
    """No answer can be produced in the current mode (e.g. a prompt missing from the replay file)."""


@dataclass(frozen=True)
class Usage:
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            model=self.model or other.model,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=round(self.cost_usd + other.cost_usd, 6),
        )


@dataclass(frozen=True)
class LLMResult[T: BaseModel]:
    output: T
    usage: Usage = field(default_factory=Usage)


class LLMClient(Protocol):
    """Anything that turns a (system, user) prompt into a validated ``schema`` instance."""

    def structured[T: BaseModel](
        self, *, task: str, system: str, user: str, schema: type[T]
    ) -> LLMResult[T]:
        """``task`` names the call site (e.g. ``"knowledge.answer"``) for tracing and replay."""
        ...


def prompt_key(task: str, system: str, user: str) -> str:
    """Stable key identifying one LLM call, used by the replay file."""
    digest = hashlib.sha256(f"{task}\x00{system}\x00{user}".encode()).hexdigest()
    return f"{task}:{digest[:24]}"


# USD per million tokens: (input, output, cache read, cache write).
PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0
) -> float:
    price = PRICING.get(model)
    if price is None:
        return 0.0
    p_in, p_out, p_read, p_write = price
    total = (
        input_tokens * p_in + output_tokens * p_out + cache_read * p_read + cache_write * p_write
    )
    return round(total / 1_000_000, 6)
