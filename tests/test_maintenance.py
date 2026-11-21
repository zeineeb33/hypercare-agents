from __future__ import annotations

import json
import sqlite3

import pytest

from hypercare.agents.maintenance import (
    KnowledgeMaintenanceAgent,
    ReviewError,
    append_section,
    bump_version,
    review_proposal,
)
from hypercare.db import query, query_one
from hypercare.llm.base import LLMError
from hypercare.llm.fake import FakeLLM
from hypercare.retrieval import KnowledgeIndex
from tests.conftest import make_llm


def test_helpers() -> None:
    assert bump_version("1.2") == "1.3"
    assert bump_version("2") == "2.1"
    assert append_section("Body\n", "New", "Text") == "Body\n\n## New\nText\n"


def test_detects_only_the_undocumented_pattern(
    conn: sqlite3.Connection, index: KnowledgeIndex, llm: FakeLLM
) -> None:
    gaps, covered, skipped, _ = KnowledgeMaintenanceAgent(conn, llm, index).detect_gaps()
    assert [g.subcategory for g in gaps] == ["INLAND_CHARGE_MISSING"]
    assert gaps[0].target_document == "guide_inland_tariff.md"
    assert len(gaps[0].ticket_ids) == 5
    assert set(covered) >= {"FEATURE_QUESTION", "RATE_MISMATCH"}
    assert skipped == []


def test_full_cycle_propose_approve(
    conn: sqlite3.Connection, index: KnowledgeIndex, llm: FakeLLM
) -> None:
    report = KnowledgeMaintenanceAgent(conn, llm, index).run()
    assert len(report.proposals) == 1
    proposal = report.proposals[0]
    checks = {c.name: c for c in proposal.checks}
    assert checks["CLOSES_THE_GAP"].passed  # the updated document answers the question
    assert checks["SOURCES_EXIST"].passed
    # The draft leaves a placeholder: it cannot be published as is.
    assert not checks["NO_PLACEHOLDER_LEFT"].passed
    assert "[owner to confirm]" in checks["NO_PLACEHOLDER_LEFT"].detail
    with pytest.raises(ReviewError, match="NO_PLACEHOLDER_LEFT"):
        review_proposal(conn, proposal.proposal_id, "APPROVED", "kb.reviewer")

    # A second run does not duplicate the pending proposal.
    again = KnowledgeMaintenanceAgent(conn, llm, index).run()
    assert again.skipped_existing == ["INLAND_CHARGE_MISSING"]
    assert again.proposals == []

    # The reviewer fills the placeholder; the checks run again on the edited text.
    filled = "1. Check the inland tariff table.\n2. Send the request to inland-pricing@northwind.example."
    revised, _ = KnowledgeMaintenanceAgent(conn, llm, index).revise(proposal.proposal_id, filled)
    assert revised.all_checks_pass

    result = review_proposal(conn, proposal.proposal_id, "APPROVED", "kb.reviewer", "ok")
    assert result["document_version"] == "1.3"
    doc = query_one(
        conn,
        "SELECT content, version FROM kb_documents WHERE file_name = ?",
        ["guide_inland_tariff.md"],
    )
    assert doc is not None
    assert "## Request inland rates that are missing" in doc["content"]
    versions = query(
        conn, "SELECT version_number, is_current FROM kb_versions ORDER BY version_number"
    )
    assert [(v["version_number"], v["is_current"]) for v in versions] == [(1, 0), (2, 1)]

    assert "inland-pricing@northwind.example" in doc["content"]

    with pytest.raises(ReviewError, match="only PENDING"):
        review_proposal(conn, proposal.proposal_id, "REJECTED", "kb.reviewer")
    with pytest.raises(ReviewError, match="only PENDING"):
        KnowledgeMaintenanceAgent(conn, llm, index).revise(proposal.proposal_id, "x")


def test_revise_unknown_proposal(
    conn: sqlite3.Connection, index: KnowledgeIndex, llm: FakeLLM
) -> None:
    with pytest.raises(ReviewError, match="not found"):
        KnowledgeMaintenanceAgent(conn, llm, index).revise(99, "x")


def test_proposal_that_does_not_close_the_gap_cannot_be_approved(
    conn: sqlite3.Connection, index: KnowledgeIndex
) -> None:
    llm = make_llm(
        maintenance__propose={
            "proposed_title": "Unrelated tips",
            "proposed_content": "Be nice to customers.",
            "rationale": "-",
            "sources_used": ["invented.md"],
        }
    )
    proposal = KnowledgeMaintenanceAgent(conn, llm, index).run().proposals[0]
    failing = {c.name for c in proposal.checks if not c.passed}
    assert failing == {"SOURCES_EXIST", "CLOSES_THE_GAP"}
    with pytest.raises(ReviewError, match="failing pre-publication checks"):
        review_proposal(conn, proposal.proposal_id, "APPROVED", "kb.reviewer")
    assert (
        review_proposal(conn, proposal.proposal_id, "REJECTED", "kb.reviewer")["status"]
        == "REJECTED"
    )


def test_model_failure_creates_no_proposal(conn: sqlite3.Connection, index: KnowledgeIndex) -> None:
    llm = make_llm(maintenance__propose=LLMError("down"))
    assert KnowledgeMaintenanceAgent(conn, llm, index).run().proposals == []


def test_review_validation(conn: sqlite3.Connection) -> None:
    with pytest.raises(ReviewError, match="APPROVED or REJECTED"):
        review_proposal(conn, 1, "MAYBE", "x")
    with pytest.raises(ReviewError, match="reviewer"):
        review_proposal(conn, 1, "APPROVED", " ")
    with pytest.raises(ReviewError, match="not found"):
        review_proposal(conn, 42, "APPROVED", "x")


def test_proposal_rows_are_json(
    conn: sqlite3.Connection, index: KnowledgeIndex, llm: FakeLLM
) -> None:
    KnowledgeMaintenanceAgent(conn, llm, index).run()
    row = query_one(conn, "SELECT related_ticket_ids, checks FROM kb_proposals")
    assert row is not None
    assert len(json.loads(row["related_ticket_ids"])) == 5
    assert len(json.loads(row["checks"])) == 4  # sources, duplicate, placeholders, gap
