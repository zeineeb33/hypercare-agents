"""Wire settings, database, retrieval index, LLM client and agents together."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from hypercare.agents.maintenance import KnowledgeMaintenanceAgent
from hypercare.agents.orchestrator import Orchestrator
from hypercare.config import Settings, get_settings
from hypercare.db import connect, query
from hypercare.llm import LLMClient, build_llm
from hypercare.retrieval import KnowledgeIndex
from hypercare.retrieval.index import Embedder, FastEmbedder
from hypercare.seed import build_database


def ensure_database(settings: Settings) -> None:
    if not settings.db_path.exists():
        build_database(settings.db_path, settings.seed_dir, settings.kb_dir)


def load_index(conn: sqlite3.Connection, embedder: Embedder | None = None) -> KnowledgeIndex:
    return KnowledgeIndex.from_documents(query(conn, "SELECT * FROM kb_documents"), embedder)


@dataclass
class Services:
    settings: Settings
    conn: sqlite3.Connection
    llm: LLMClient
    index: KnowledgeIndex
    embedder: Embedder | None = None

    def refresh_index(self) -> None:
        """Rebuild the index after the knowledge base changed (e.g. an approved proposal)."""
        self.index = load_index(self.conn, self.embedder)

    def orchestrator(self) -> Orchestrator:
        return Orchestrator(
            self.conn,
            self.llm,
            self.index,
            top_k=self.settings.retrieval_top_k,
            recurring_min_tickets=self.settings.recurring_min_tickets,
        )

    def maintenance(self) -> KnowledgeMaintenanceAgent:
        return KnowledgeMaintenanceAgent(
            self.conn,
            self.llm,
            self.index,
            min_tickets=self.settings.recurring_min_tickets,
            top_k=self.settings.retrieval_top_k,
        )


def open_services(
    settings: Settings | None = None, *, record: bool = False, llm: LLMClient | None = None
) -> Services:
    settings = settings or get_settings()
    ensure_database(settings)
    conn = connect(settings.db_path)
    embedder: Embedder | None = FastEmbedder() if settings.retrieval == "hybrid" else None
    return Services(
        settings=settings,
        conn=conn,
        llm=llm or build_llm(settings, record=record),
        index=load_index(conn, embedder),
        embedder=embedder,
    )
