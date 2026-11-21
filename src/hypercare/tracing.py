"""Step-level tracing: every agent step is timed, costed and persisted to ``agent_traces``."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from hypercare.agents.schemas import StepTrace
from hypercare.db import transaction, utcnow
from hypercare.llm.base import Usage


def new_execution_id(prefix: str = "RUN") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@dataclass
class StepHandle:
    """Filled in by the code inside a ``Tracer.step`` block."""

    summary: str = ""
    decision: str | None = None
    sources: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    input_summary: str = ""
    skipped: bool = False

    def add_usage(self, usage: Usage) -> None:
        self.usage = self.usage + usage


class Tracer:
    def __init__(
        self, conn: sqlite3.Connection, execution_id: str, ticket_id: int | None = None
    ) -> None:
        self.conn = conn
        self.execution_id = execution_id
        self.ticket_id = ticket_id
        self.steps: list[StepTrace] = []
        self.usage = Usage()

    @contextmanager
    def step(self, agent: str, action: str) -> Iterator[StepHandle]:
        handle = StepHandle()
        start = time.perf_counter()
        status = "OK"
        try:
            yield handle
        except Exception as e:
            status = "ERROR"
            handle.summary = handle.summary or f"{type(e).__name__}: {e}"
            raise
        finally:
            if handle.skipped and status == "OK":
                status = "SKIPPED"
            duration = int((time.perf_counter() - start) * 1000)
            self.usage = self.usage + handle.usage
            self.steps.append(
                StepTrace(
                    agent=agent,
                    action=action,
                    status=status,  # type: ignore[arg-type]
                    duration_ms=duration,
                    summary=handle.summary[:300],
                    cost_usd=handle.usage.cost_usd,
                )
            )
            with transaction(self.conn):
                self.conn.execute(
                    "INSERT INTO agent_traces (execution_id, ticket_id, created_at, agent, action, "
                    "input_summary, output_summary, sources, decision, status, duration_ms, "
                    "input_tokens, output_tokens, cost_usd) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        self.execution_id,
                        self.ticket_id,
                        utcnow(),
                        agent,
                        action,
                        handle.input_summary[:2000],
                        handle.summary[:4000],
                        json.dumps(handle.sources),
                        handle.decision,
                        status,
                        duration,
                        handle.usage.input_tokens,
                        handle.usage.output_tokens,
                        handle.usage.cost_usd,
                    ),
                )
