"""Scenario-based evaluation of the whole pipeline.

Each case names a ticket, the acceptable decisions, the documents an answer should cite and the
facts it should state. Cases run on a private copy of the database, in parallel.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

from hypercare.agents.orchestrator import Orchestrator
from hypercare.agents.schemas import TicketRun
from hypercare.agents.verification import numbers_in
from hypercare.config import DATA_DIR
from hypercare.db import connect
from hypercare.llm.base import LLMClient
from hypercare.retrieval import KnowledgeIndex

CASES_PATH = DATA_DIR / "eval" / "cases.json"


class EvalCase(BaseModel):
    ticket_id: int
    expected: list[str]
    sources: list[str] = []
    mention: list[str] = []
    note: str = ""


class CaseResult(BaseModel):
    ticket_id: int
    ticket_ref: str
    note: str
    expected: list[str]
    proposed: str | None
    actual: str
    decision_ok: bool
    sources_ok: bool | None
    mentions_ok: bool | None
    missing_mentions: list[str]
    overridden: bool
    duration_ms: int
    cost_usd: float
    response: str
    error: str | None = None


class EvalReport(BaseModel):
    model: str
    cases: int
    decision_accuracy: float
    source_accuracy: float | None
    fact_accuracy: float | None
    overrides: int
    errors: int
    median_latency_s: float
    total_cost_usd: float
    cost_per_ticket_usd: float
    confusion: dict[str, dict[str, int]]
    results: list[CaseResult]


def load_cases(path: Path = CASES_PATH) -> list[EvalCase]:
    return [EvalCase.model_validate(c) for c in json.loads(path.read_text())]


def mentions(text: str, token: str) -> bool:
    """``token`` may list alternatives with ``|``; digits are matched as numbers (1,850 = 1850)."""
    for raw in token.split("|"):
        alt = raw.strip()
        if alt.isdigit() and int(alt) in numbers_in(text):
            return True
        if not alt.isdigit() and alt.lower() in text.lower():
            return True
    return False


def score(case: EvalCase, run: TicketRun) -> CaseResult:
    actual = str(run.final.decision)
    decision_ok = actual in case.expected
    answered = actual == "ANSWER"
    sources_ok = mentions_ok = None
    missing: list[str] = []
    if answered and case.sources:
        sources_ok = any(s in run.final.sources for s in case.sources)
    if answered and case.mention:
        missing = [m for m in case.mention if not mentions(run.final.response, m)]
        mentions_ok = not missing
    return CaseResult(
        ticket_id=case.ticket_id,
        ticket_ref=run.ticket_ref,
        note=case.note,
        expected=case.expected,
        proposed=str(run.proposed.decision) if run.proposed else None,
        actual=actual,
        decision_ok=decision_ok,
        sources_ok=sources_ok,
        mentions_ok=mentions_ok,
        missing_mentions=missing,
        overridden=bool(run.verification and run.verification.overridden),
        duration_ms=run.duration_ms,
        cost_usd=run.cost_usd,
        response=run.final.response,
    )


def _ratio(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(model: str, results: list[CaseResult]) -> EvalReport:
    confusion: dict[str, Counter[str]] = {}
    for r in results:
        confusion.setdefault("/".join(r.expected), Counter())[r.actual] += 1
    total_cost = round(sum(r.cost_usd for r in results), 4)
    return EvalReport(
        model=model,
        cases=len(results),
        decision_accuracy=_ratio([r.decision_ok for r in results]) or 0.0,
        source_accuracy=_ratio([r.sources_ok for r in results if r.sources_ok is not None]),
        fact_accuracy=_ratio([r.mentions_ok for r in results if r.mentions_ok is not None]),
        overrides=sum(r.overridden for r in results),
        errors=sum(r.error is not None for r in results),
        median_latency_s=round(statistics.median(r.duration_ms for r in results) / 1000, 1)
        if results
        else 0.0,
        total_cost_usd=total_cost,
        cost_per_ticket_usd=round(total_cost / len(results), 4) if results else 0.0,
        confusion={k: dict(v) for k, v in confusion.items()},
        results=sorted(results, key=lambda r: r.ticket_id),
    )


def run_evaluation(
    source_conn: sqlite3.Connection,
    llm: LLMClient,
    index: KnowledgeIndex,
    *,
    model: str,
    cases: list[EvalCase] | None = None,
    workers: int = 4,
    recurring_min_tickets: int = 4,
    top_k: int = 4,
) -> EvalReport:
    cases = cases if cases is not None else load_cases()
    with tempfile.TemporaryDirectory() as tmp:
        db_copy = Path(tmp) / "eval.db"
        target = sqlite3.connect(db_copy)
        source_conn.backup(target)
        target.close()

        def one(case: EvalCase) -> CaseResult:
            conn = connect(db_copy)
            conn.execute("PRAGMA busy_timeout = 30000")
            try:
                orchestrator = Orchestrator(
                    conn, llm, index, top_k=top_k, recurring_min_tickets=recurring_min_tickets
                )
                return score(case, orchestrator.run(case.ticket_id))
            except Exception as e:
                return CaseResult(
                    ticket_id=case.ticket_id,
                    ticket_ref="",
                    note=case.note,
                    expected=case.expected,
                    proposed=None,
                    actual="ERROR",
                    decision_ok=False,
                    sources_ok=None,
                    mentions_ok=None,
                    missing_mentions=[],
                    overridden=False,
                    duration_ms=0,
                    cost_usd=0.0,
                    response="",
                    error=f"{type(e).__name__}: {e}",
                )
            finally:
                conn.close()

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            results = list(pool.map(one, cases))
    return summarize(model, results)


def to_markdown(report: EvalReport) -> str:
    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.0%}"

    lines = [
        f"# Evaluation report ({report.model})",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Cases | {report.cases} |",
        f"| Decision accuracy | {pct(report.decision_accuracy)} |",
        f"| Answers citing the expected document | {pct(report.source_accuracy)} |",
        f"| Answers stating the expected facts | {pct(report.fact_accuracy)} |",
        f"| Decisions overridden by verification | {report.overrides} |",
        f"| Errors | {report.errors} |",
        *(
            [
                f"| Median latency per ticket | {report.median_latency_s} s |",
                f"| Cost per ticket | ${report.cost_per_ticket_usd:.4f} |",
            ]
            if report.total_cost_usd > 0
            else ["| Latency and cost | n/a: answers replayed, not timed or billed |"]
        ),
        "",
        "| Ticket | Case | Expected | Proposed | Final | OK |",
        "|---|---|---|---|---|---|",
    ]
    for r in report.results:
        ok = "yes" if r.decision_ok else "**no**"
        if r.missing_mentions:
            ok += f" (missing: {', '.join(r.missing_mentions)})"
        lines.append(
            f"| {r.ticket_ref} | {r.note} | {' / '.join(r.expected)} | {r.proposed or '-'} | "
            f"{r.actual} | {ok} |"
        )
    return "\n".join(lines) + "\n"
