from __future__ import annotations

import sqlite3

from hypercare.agents.knowledge import KnowledgeAgent
from hypercare.agents.orchestrator import Orchestrator
from hypercare.agents.resolution import ResolutionAgent, build_prompt
from hypercare.agents.schemas import Decision
from hypercare.db import query
from hypercare.llm.base import LLMError
from hypercare.llm.fake import FakeLLM
from hypercare.retrieval import KnowledgeIndex
from tests.conftest import make_llm

# ---------------------------------------------------------------- knowledge agent


def test_knowledge_agent_answers_from_retrieved_passages(
    index: KnowledgeIndex, llm: FakeLLM
) -> None:
    result, usage = KnowledgeAgent(llm, index).run("Can I put hazardous cargo on a SOC line?")
    assert result.answer is not None
    assert not result.insufficient
    assert "rules_special_cargo.md" in result.retrieved_files
    assert [c.task for c in llm.calls] == ["knowledge.plan", "knowledge.answer"]
    assert usage.input_tokens == 20


def test_knowledge_agent_survives_planning_failure(index: KnowledgeIndex) -> None:
    llm = make_llm(knowledge__plan=LLMError("down"))
    result, _ = KnowledgeAgent(llm, index).run("What is the difference between BAF and LSS?")
    assert result.queries == []
    assert result.answer is not None  # retrieval fell back to the raw question


def test_knowledge_agent_without_hits_skips_the_model(index: KnowledgeIndex, llm: FakeLLM) -> None:
    result, _ = KnowledgeAgent(llm, index).run("zzz qqq")
    assert result.insufficient
    assert "knowledge.answer" not in [c.task for c in llm.calls]


def test_knowledge_agent_records_answer_failure(index: KnowledgeIndex) -> None:
    llm = make_llm(knowledge__answer=LLMError("refused"))
    result, _ = KnowledgeAgent(llm, index).run("What does SOC mean?")
    assert result.error == "refused"
    assert result.insufficient


def test_related_document_is_pinned(index: KnowledgeIndex, llm: FakeLLM) -> None:
    agent = KnowledgeAgent(llm, index, top_k=2)
    hits = agent.retrieve("surcharges", [], related_doc="faq_hypercare.md")
    assert hits[0].passage.file_name == "faq_hypercare.md"


# ---------------------------------------------------------------- resolution agent


def test_resolution_prompt_mentions_failures() -> None:
    assert "not run" in build_prompt("t", None, None)


def test_resolution_failure_escalates() -> None:
    resolution, _ = ResolutionAgent(make_llm(resolution__decide=LLMError("timeout"))).run(
        "t", None, None
    )
    assert resolution.decision == Decision.ESCALATE
    assert "timeout" in resolution.response


# ---------------------------------------------------------------- orchestrator


def orchestrator(conn: sqlite3.Connection, index: KnowledgeIndex, llm: FakeLLM) -> Orchestrator:
    return Orchestrator(conn, llm, index)


def test_answer_with_data(conn: sqlite3.Connection, index: KnowledgeIndex, llm: FakeLLM) -> None:
    run = orchestrator(conn, index, llm).run(3)
    assert run.final.decision == Decision.ANSWER
    assert run.verification is not None
    assert run.verification.passed_count == 6
    assert [s.agent for s in run.steps][:4] == [
        "ORCHESTRATOR",
        "KNOWLEDGE_AGENT",
        "DATA_INVESTIGATION_AGENT",
        "ORCHESTRATOR",
    ]
    traces = query(
        conn, "SELECT agent FROM agent_traces WHERE execution_id = ?", [run.execution_id]
    )
    assert len(traces) == len(run.steps)
    assert run.input_tokens > 0


def test_clarification_needs_no_model(
    conn: sqlite3.Connection, index: KnowledgeIndex, llm: FakeLLM
) -> None:
    run = orchestrator(conn, index, llm).run(23)
    assert run.final.decision == Decision.ASK_CLARIFICATION
    assert llm.calls == []
    assert run.final.missing_information


def test_draft_line_answer_is_overridden(conn: sqlite3.Connection, index: KnowledgeIndex) -> None:
    llm = make_llm(
        resolution__decide={
            "decision": "ANSWER",
            "response": "Confirmed, you can send USD 2600.",
            "evidence_used": [],
            "sources": ["rules_price_line_lifecycle.md"],
            "missing_information": [],
            "confidence": "HIGH",
        }
    )
    run = orchestrator(conn, index, llm).run(6)
    assert run.proposed is not None
    assert run.proposed.decision == Decision.ANSWER
    assert run.final.decision == Decision.ESCALATE
    assert "DRAFT" in run.final.response


def test_recurring_undocumented_issue_is_a_kb_gap(
    conn: sqlite3.Connection, index: KnowledgeIndex, llm: FakeLLM
) -> None:
    run = orchestrator(conn, index, llm).run(9)
    assert run.pattern is not None
    assert run.pattern.kb_gap
    assert run.final.decision == Decision.DETECT_KB_GAP
    assert "resolution.decide" not in [c.task for c in llm.calls]


def test_recurring_but_documented_is_not_a_gap(
    conn: sqlite3.Connection, index: KnowledgeIndex, llm: FakeLLM
) -> None:
    run = orchestrator(conn, index, llm).run(1)  # 5 open FEATURE_QUESTION tickets, documented
    assert run.pattern is not None
    assert run.pattern.is_recurring
    assert not run.pattern.kb_gap
    assert run.final.decision != Decision.DETECT_KB_GAP
    assert "resolution.decide" in [c.task for c in llm.calls]


def test_documentation_question_skips_data(
    conn: sqlite3.Connection, index: KnowledgeIndex, llm: FakeLLM
) -> None:
    run = orchestrator(conn, index, llm).run(26)
    assert run.data is None
    assert any(s.status == "SKIPPED" for s in run.steps)


def test_verification_crash_fails_closed(
    conn: sqlite3.Connection, index: KnowledgeIndex, llm: FakeLLM
) -> None:
    orch = orchestrator(conn, index, llm)

    def boom(*_args: object) -> None:
        raise RuntimeError("bug")

    orch.verification_agent.run = boom  # type: ignore[method-assign,assignment]
    run = orch.run(3)
    assert run.final.decision == Decision.ESCALATE
    assert run.verification is not None
    assert run.verification.checks[0].name == "VERIFICATION_ERROR"


def test_unknown_ticket(conn: sqlite3.Connection, index: KnowledgeIndex, llm: FakeLLM) -> None:
    import pytest

    with pytest.raises(LookupError):
        orchestrator(conn, index, llm).run(999)
