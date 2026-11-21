"""Deterministic stand-in for tests: scripted answers, no network."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from hypercare.llm.base import LLMError, LLMResult, Usage

Handler = Callable[[str, str], dict[str, Any] | BaseModel | Exception]


@dataclass
class Call:
    task: str
    system: str
    user: str


@dataclass
class FakeLLM:
    """Maps a task name to a handler ``(system, user) -> output``.

    A handler may return a dict, a model instance or an exception to raise.
    Every call is recorded in ``calls``.
    """

    handlers: dict[str, Handler] = field(default_factory=dict)
    calls: list[Call] = field(default_factory=list)

    def on(self, task: str, output: dict[str, Any] | BaseModel | Exception | Handler) -> FakeLLM:
        if callable(output):
            self.handlers[task] = output
        else:

            def constant(_system: str, _user: str) -> dict[str, Any] | BaseModel | Exception:
                return output

            self.handlers[task] = constant
        return self

    def structured[T: BaseModel](
        self, *, task: str, system: str, user: str, schema: type[T]
    ) -> LLMResult[T]:
        self.calls.append(Call(task, system, user))
        handler = self.handlers.get(task)
        if handler is None:
            raise LLMError(f"FakeLLM: no handler for task {task!r}")
        out = handler(system, user)
        if isinstance(out, Exception):
            raise out
        output = out if isinstance(out, schema) else schema.model_validate(out)
        return LLMResult(output=output, usage=Usage(model="fake", input_tokens=10, output_tokens=5))
