"""Claude through the official Anthropic SDK, with validated structured outputs."""

from __future__ import annotations

from typing import Any, cast

import anthropic
from anthropic.types.beta import BetaOutputConfigParam
from pydantic import BaseModel, ValidationError

from hypercare.llm.base import LLMError, LLMResult, Usage, estimate_cost

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicClient:
    """Calls ``messages.parse`` so every response is validated against a Pydantic schema.

    - Adaptive thinking is always on for Claude Opus 5.5; ``effort`` controls its depth.
    - System prompts are marked cacheable: they are identical across tickets.
    - Server-side refusal fallbacks are enabled (``fallbacks="default"``), so a policy decline is
      retried on a fallback model inside the same call.
    - Any failure raises :class:`LLMError`; the agents then fail closed.
    """

    def __init__(
        self,
        *,
        model: str,
        effort: str = "medium",
        max_tokens: int = 16000,
        api_key: str | None = None,
        client: anthropic.Anthropic | None = None,
        use_fallbacks: bool = True,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.use_fallbacks = use_fallbacks
        self._client = client or anthropic.Anthropic(api_key=api_key, max_retries=3)

    def structured[T: BaseModel](
        self, *, task: str, system: str, user: str, schema: type[T]
    ) -> LLMResult[T]:
        kwargs: dict[str, Any] = {}
        if self.use_fallbacks:
            kwargs = {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        try:
            response = self._client.beta.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                output_config=cast(BetaOutputConfigParam, {"effort": self.effort}),
                **kwargs,
            )
        except anthropic.AuthenticationError as e:
            raise LLMError("authentication failed: check ANTHROPIC_API_KEY") from e
        except anthropic.RateLimitError as e:
            raise LLMError("rate limited by the API after retries") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"API error {e.status_code} on {task}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"network error on {task}") from e
        except ValidationError as e:
            # The SDK validates the JSON before we can read stop_reason: an empty, truncated or
            # refused output surfaces here.
            raise LLMError(
                f"invalid structured output on {task} (empty, truncated or refused)"
            ) from e

        if response.stop_reason == "refusal":
            raise LLMError(f"the model declined the request on {task}")
        if response.stop_reason == "max_tokens":
            raise LLMError(f"output truncated (max_tokens) on {task}")
        parsed = response.parsed_output
        if parsed is None:
            raise LLMError(f"no structured output returned on {task}")

        u = response.usage
        cache_read = u.cache_read_input_tokens or 0
        cache_write = u.cache_creation_input_tokens or 0
        usage = Usage(
            model=response.model,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            cost_usd=estimate_cost(
                response.model, u.input_tokens, u.output_tokens, cache_read, cache_write
            ),
        )
        return LLMResult(output=parsed, usage=usage)
