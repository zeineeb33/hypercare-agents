"""Record real LLM calls once, replay them for free.

Only the LLM calls are replayed: retrieval, SQL investigation and verification still run for real.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from hypercare.llm.base import LLMClient, LLMResult, LLMUnavailableError, Usage, prompt_key


def load_replays(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data: dict[str, dict[str, Any]] = json.loads(path.read_text())
    return data


class ReplayClient:
    """Serves recorded outputs. A prompt that was never recorded raises LLMUnavailableError."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries = load_replays(path)

    def __len__(self) -> int:
        return len(self._entries)

    def structured[T: BaseModel](
        self, *, task: str, system: str, user: str, schema: type[T]
    ) -> LLMResult[T]:
        key = prompt_key(task, system, user)
        entry = self._entries.get(key)
        if entry is None:
            raise LLMUnavailableError(
                f"replay mode: no recorded answer for {task}; run in live mode to process it"
            )
        try:
            output = schema.model_validate(entry["output"])
        except ValidationError as e:
            raise LLMUnavailableError(f"replay entry for {task} does not match the schema") from e
        return LLMResult(output=output, usage=Usage(**entry["usage"]))


class RecordingClient:
    """Wraps a live client and stores every successful call in the replay file."""

    def __init__(self, inner: LLMClient, path: Path) -> None:
        self.inner = inner
        self.path = path
        self._entries = load_replays(path)
        self._lock = threading.Lock()

    def structured[T: BaseModel](
        self, *, task: str, system: str, user: str, schema: type[T]
    ) -> LLMResult[T]:
        result = self.inner.structured(task=task, system=system, user=user, schema=schema)
        with self._lock:
            self._entries[prompt_key(task, system, user)] = {
                "task": task,
                "output": result.output.model_dump(mode="json"),
                "usage": asdict(result.usage),
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._entries, indent=1, sort_keys=True) + "\n")
        return result
