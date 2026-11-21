"""Produce replays without the API: collect prompts, answer them elsewhere, import the answers.

Used to record the demo with Claude through Claude Code instead of paid API calls:

1. ``hypercare prompts export`` runs the pipeline with :class:`CollectingClient`. Prompts that
   already have a recorded answer are replayed; the others are written to a pending file.
2. Each pending prompt is answered (as JSON matching its schema) in a separate answers file.
3. ``hypercare prompts import`` validates the answers and adds them to the replay file.

Steps 1 to 3 repeat until nothing is pending, because each agent waits for the previous one.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from hypercare.agents.schemas import (
    KnowledgeAnswerOutput,
    ProposalDraftOutput,
    ResolutionOutput,
    SearchPlanOutput,
)
from hypercare.llm.base import LLMResult, LLMUnavailableError, Usage, prompt_key
from hypercare.llm.replay import ReplayClient, load_replays

SCHEMAS: dict[str, type[BaseModel]] = {
    "SearchPlanOutput": SearchPlanOutput,
    "KnowledgeAnswerOutput": KnowledgeAnswerOutput,
    "ResolutionOutput": ResolutionOutput,
    "ProposalDraftOutput": ProposalDraftOutput,
}


class CollectingClient:
    """Replays known prompts; records unknown ones as pending and raises LLMUnavailableError."""

    def __init__(self, replay_path: Path) -> None:
        self.replay = ReplayClient(replay_path)
        self.pending: dict[str, dict[str, Any]] = {}
        self._blocked = False

    def start_run(self) -> None:
        """Call before each independent run (one ticket, one maintenance pass)."""
        self._blocked = False

    def structured[T: BaseModel](
        self, *, task: str, system: str, user: str, schema: type[T]
    ) -> LLMResult[T]:
        if self._blocked:
            # An earlier prompt of this run is pending: later prompts would be built on fallback
            # paths and never match the real run, so they are not collected.
            raise LLMUnavailableError(f"{task}: waiting for an earlier pending answer")
        try:
            return self.replay.structured(task=task, system=system, user=user, schema=schema)
        except LLMUnavailableError:
            self._blocked = True
            key = prompt_key(task, system, user)
            self.pending[key] = {
                "task": task,
                "schema": schema.__name__,
                "json_schema": schema.model_json_schema(),
                "system": system,
                "user": user,
            }
            raise


class AnswerImportError(Exception):
    pass


def import_answers(
    pending_path: Path, answers_path: Path, replay_path: Path, *, model_label: str
) -> int:
    """Validate answers against the pending prompts' schemas and add them to the replay file."""
    pending: dict[str, dict[str, Any]] = json.loads(pending_path.read_text())
    answers: dict[str, Any] = json.loads(answers_path.read_text())
    unknown = sorted(set(answers) - set(pending))
    if unknown:
        raise AnswerImportError(f"answers for prompts that are not pending: {unknown[:3]}")
    entries = load_replays(replay_path)
    for key, output in answers.items():
        item = pending[key]
        schema = SCHEMAS[item["schema"]]
        try:
            validated = schema.model_validate(output)
        except ValidationError as e:
            raise AnswerImportError(f"{key}: answer does not match {item['schema']}: {e}") from e
        entries[key] = {
            "task": item["task"],
            "output": validated.model_dump(mode="json"),
            "usage": asdict(Usage(model=model_label)),
        }
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(json.dumps(entries, indent=1, sort_keys=True) + "\n")
    return len(answers)
