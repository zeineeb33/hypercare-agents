from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from hypercare.agents.data_investigation import DataInvestigationAgent
from hypercare.agents.schemas import (
    DataEvidence,
    Decision,
    KnowledgeAnswerOutput,
    KnowledgeResult,
    PassageRef,
    Resolution,
)
from hypercare.agents.verification import VerificationAgent, failed_closed, numbers_in


def knowledge(insufficient: bool = False, files: tuple[str, ...] = ("doc.md",)) -> KnowledgeResult:
    return KnowledgeResult(
        question="q",
        passages=[PassageRef(file_name=f, section="s", version="1", score=1.0) for f in files],
        answer=KnowledgeAnswerOutput(
            answer="a",
            sources_used=list(files),
            confidence="HIGH",
            insufficient_evidence=insufficient,
        ),
    )


def answer(text: str, sources: list[str] | None = None) -> Resolution:
    return Resolution(decision=Decision.ANSWER, response=text, sources=sources or ["doc.md"])


def check(result: Any, name: str) -> Any:
    return next(c for c in result.checks if c.name == name)


@pytest.fixture
def evidence(conn: sqlite3.Connection) -> Any:
    agent = DataInvestigationAgent(conn)
    return agent.run


def test_numbers_in() -> None:
    assert numbers_in("USD 1,850 then 2 150 and 2150.") >= {1850, 2150}


def test_clean_answer_passes(evidence: Any) -> None:
    r = VerificationAgent().run(
        answer("Base 1,850 plus surcharges gives 2,150.", ["doc.md", "QS-10231"]),
        knowledge(),
        evidence(3),
    )
    assert r.verified_decision == Decision.ANSWER
    assert r.passed_count == 6
    assert not r.overridden


def test_draft_line_is_escalated(evidence: Any) -> None:
    r = VerificationAgent().run(answer("Confirmed: 2600."), knowledge(), evidence(6))
    assert r.verified_decision == Decision.ESCALATE
    assert not check(r, "DRAFT_LINE_ANSWERED").passed


def test_overlap_is_escalated(evidence: Any) -> None:
    r = VerificationAgent().run(answer("Use line 2 at 1800."), knowledge(), evidence(8))
    assert not check(r, "OVERLAPPING_LINES_ANSWERED").passed
    assert r.verified_decision == Decision.ESCALATE


def test_expired_rate_presented_as_current(evidence: Any) -> None:
    bad = VerificationAgent().run(answer("The rate is 1950."), knowledge(), evidence(4))
    assert not check(bad, "EXPIRED_RATE_AS_CURRENT").passed
    good = VerificationAgent().run(
        answer("1950 was the expired line; the current rate is 2100."), knowledge(), evidence(4)
    )
    assert check(good, "EXPIRED_RATE_AS_CURRENT").passed


def test_kb_insufficient_without_data() -> None:
    r = VerificationAgent().run(answer("Here you go."), knowledge(insufficient=True), None)
    assert not check(r, "KB_INSUFFICIENT_BUT_ANSWERED").passed
    assert r.verified_decision == Decision.ESCALATE


def test_kb_insufficient_but_backed_by_data(evidence: Any) -> None:
    r = VerificationAgent().run(
        answer("Total 4300 = 2 x 2150.", ["BK-55012"]), knowledge(insufficient=True), evidence(16)
    )
    assert check(r, "KB_INSUFFICIENT_BUT_ANSWERED").passed


def test_unretrieved_document_cited() -> None:
    r = VerificationAgent().run(answer("x", ["invented.md"]), knowledge(), None)
    assert not check(r, "UNSUPPORTED_SOURCES").passed
    assert r.verified_decision == Decision.ESCALATE


def test_no_source_at_all() -> None:
    r = VerificationAgent().run(answer("x", ["general knowledge"]), knowledge(), None)
    assert not check(r, "UNSUPPORTED_SOURCES").passed


def test_missing_amount_is_flagged_without_override(evidence: Any) -> None:
    r = VerificationAgent().run(
        answer("It is all-in.", ["doc.md", "QS-10231"]), knowledge(), evidence(3)
    )
    c = check(r, "RATE_NOT_IN_RESPONSE")
    assert not c.passed
    assert not c.overrides
    assert r.verified_decision == Decision.ANSWER


def test_non_answers_are_not_checked() -> None:
    esc = Resolution(decision=Decision.ESCALATE, response="Escalated")
    data = DataEvidence(
        ticket={}, price_lines=[{"status": "DRAFT", "pricelist_code": "P", "line_ref": 1}]
    )
    r = VerificationAgent().run(esc, None, data)
    assert all(c.passed for c in r.checks)
    assert r.verified_decision == Decision.ESCALATE


def test_failed_closed() -> None:
    r = failed_closed(answer("x"), "boom")
    assert r.verified_decision == Decision.ESCALATE
    assert r.overridden
