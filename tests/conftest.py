from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from hypercare.config import DATA_DIR
from hypercare.db import connect
from hypercare.llm.fake import FakeLLM
from hypercare.retrieval import KnowledgeIndex
from hypercare.seed import build_database
from hypercare.services import load_index

SEED_DIR = DATA_DIR / "seed"
KB_DIR = DATA_DIR / "kb"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    build_database(path, SEED_DIR, KB_DIR)
    return path


@pytest.fixture
def conn(db_path: Path) -> Iterator[sqlite3.Connection]:
    c = connect(db_path)
    yield c
    c.close()


@pytest.fixture
def index(conn: sqlite3.Connection) -> KnowledgeIndex:
    return load_index(conn)


def files_in_prompt(user: str) -> list[str]:
    return sorted(set(re.findall(r"\[([\w.]+\.md) \|", user)))


def question_of(user: str) -> str:
    return user.rsplit("QUESTION:\n", 1)[-1].lower()


def knowledge_answer(_system: str, user: str) -> dict[str, Any]:
    """A plausible knowledge agent: inland 'how to add' questions are not covered by the docs."""
    q = question_of(user)
    files = files_in_prompt(user)
    gap_question = "inland" in q or "pre-carriage" in q or "inl00" in q
    covered = "request inland rates" in user.lower()  # a proposal added the missing section
    if gap_question and not covered:
        return {
            "answer": "The inland tariff guide explains how to check an inland rate but not how to "
            "get a missing rate added.",
            "sources_used": files[:1],
            "confidence": "MEDIUM",
            "insufficient_evidence": True,
        }
    return {
        "answer": "Documented answer.",
        "sources_used": files[:1],
        "confidence": "HIGH",
        "insufficient_evidence": False,
    }


def make_llm(**overrides: Any) -> FakeLLM:
    llm = FakeLLM()
    llm.on("knowledge.plan", lambda _s, u: {"queries": [u.splitlines()[-1][:60]]})
    llm.on("knowledge.answer", knowledge_answer)
    llm.on(
        "resolution.decide",
        {
            "decision": "ANSWER",
            "response": "The all-in price is USD 2,150: base rate 1,850 plus BAF 220 and LSS 80.",
            "evidence_used": ["QS-10231 quoted 2150"],
            "sources": ["rules_all_in_vs_base_rate.md", "QS-10231"],
            "missing_information": [],
            "confidence": "HIGH",
        },
    )
    llm.on(
        "maintenance.propose",
        {
            "proposed_title": "Request inland rates that are missing",
            "proposed_content": "1. Check the inland tariff table.\n2. If no INL00 rate exists, "
            "send the inland point and gateway port to [owner to confirm].",
            "rationale": "Five open tickets ask how to get a missing inland rate added.",
            "sources_used": ["guide_inland_tariff.md"],
        },
    )
    for task, output in overrides.items():
        llm.on(task.replace("__", "."), output)
    return llm


@pytest.fixture
def llm() -> FakeLLM:
    return make_llm()
