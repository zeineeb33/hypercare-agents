"""Self-healing knowledge base: detect recurring gaps, draft an update, test it, let a human approve.

1. ``detect_gaps``: subcategories with many open tickets whose representative question the
   knowledge agent cannot answer.
2. ``propose``: the model drafts a new section for the most relevant document, then
   pre-publication checks run, including re-asking the question against the updated document.
3. ``review``: a human approves or rejects; approval versions the document and publishes it.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass

from pydantic import BaseModel

from hypercare.agents.data_investigation import DataInvestigationAgent
from hypercare.agents.knowledge import KnowledgeAgent
from hypercare.agents.resolution import _compact
from hypercare.agents.schemas import KnowledgeResult, ProposalDraftOutput, VerificationCheck
from hypercare.db import query, query_one, transaction, utcnow
from hypercare.llm.base import LLMClient, LLMError, Usage
from hypercare.retrieval import KnowledgeIndex

PROPOSAL_SYSTEM = """\
You maintain the knowledge base of the hypercare support team for TideRate (pricing platform) and
QuotePort (quoting portal) at Northwind Shipping. Several open tickets ask the same question and the
current documentation does not answer it.

Write one new section to append to the target document so that the question is answered.
- Practical and step by step, in the style of the target document (Markdown, no top-level heading).
- Use only facts found in the provided documents and data. When a step needs a fact the sources do
  not give (an owner, a contact, a lead time), write a placeholder in square brackets for the
  reviewer to fill, for example [contact to confirm].
- Do not contradict the existing documents.
- List in sources_used the file names of the documents you relied on."""


class GapCandidate(BaseModel):
    subcategory: str
    ticket_ids: list[int]
    ticket_refs: list[str]
    representative_ticket_id: int
    question: str
    target_document: str | None
    knowledge: KnowledgeResult


class ProposalResult(BaseModel):
    proposal_id: int
    subcategory: str
    target_document: str
    proposed_title: str
    checks: list[VerificationCheck]

    @property
    def all_checks_pass(self) -> bool:
        return all(c.passed for c in self.checks)


@dataclass
class MaintenanceReport:
    candidates: int
    covered: list[str]
    skipped_existing: list[str]
    proposals: list[ProposalResult]
    usage: Usage


def bump_version(version: str) -> str:
    major, _, minor = version.partition(".")
    return f"{major}.{int(minor or 0) + 1}" if minor.isdigit() or not minor else f"{version}.1"


def append_section(content: str, title: str, body: str) -> str:
    return f"{content.rstrip()}\n\n## {title.strip()}\n{body.strip()}\n"


class ReviewError(Exception):
    pass


class KnowledgeMaintenanceAgent:
    def __init__(
        self,
        conn: sqlite3.Connection,
        llm: LLMClient,
        index: KnowledgeIndex,
        *,
        min_tickets: int = 4,
        top_k: int = 4,
    ) -> None:
        self.conn = conn
        self.llm = llm
        self.index = index
        self.min_tickets = min_tickets
        self.top_k = top_k
        self.knowledge_agent = KnowledgeAgent(llm, index, top_k)

    # ------------------------------------------------------------------ detection

    def _has_open_proposal(self, subcategory: str) -> bool:
        row = query_one(
            self.conn,
            "SELECT 1 FROM kb_proposals WHERE subcategory = ? AND status IN ('PENDING', 'APPROVED')",
            [subcategory],
        )
        return row is not None

    def recurring_subcategories(self) -> list[tuple[str, list[dict[str, object]]]]:
        groups: dict[str, list[dict[str, object]]] = {}
        for t in query(
            self.conn,
            "SELECT ticket_id, ticket_ref, subcategory, description, related_doc FROM tickets "
            "WHERE status != 'RESOLVED' ORDER BY created_at",
        ):
            groups.setdefault(str(t["subcategory"]), []).append(t)
        return [(s, ts) for s, ts in groups.items() if len(ts) >= self.min_tickets]

    def detect_gaps(self) -> tuple[list[GapCandidate], list[str], list[str], Usage]:
        """Return (gaps, covered subcategories, subcategories with an existing proposal, usage)."""
        gaps: list[GapCandidate] = []
        covered: list[str] = []
        skipped: list[str] = []
        usage = Usage()
        for subcategory, tickets in self.recurring_subcategories():
            if self._has_open_proposal(subcategory):
                skipped.append(subcategory)
                continue
            first = tickets[0]
            docs = Counter(str(t["related_doc"]) for t in tickets if t["related_doc"])
            target = docs.most_common(1)[0][0] if docs else None
            knowledge, u = self.knowledge_agent.run(str(first["description"]), target)
            usage = usage + u
            if knowledge.answer is None or not knowledge.insufficient:
                covered.append(subcategory)  # answered, or unknown: no gap claimed
                continue
            if target is None and knowledge.passages:
                target = knowledge.passages[0].file_name
            gaps.append(
                GapCandidate(
                    subcategory=subcategory,
                    ticket_ids=[int(str(t["ticket_id"])) for t in tickets],
                    ticket_refs=[str(t["ticket_ref"]) for t in tickets],
                    representative_ticket_id=int(str(first["ticket_id"])),
                    question=str(first["description"]),
                    target_document=target,
                    knowledge=knowledge,
                )
            )
        return gaps, covered, skipped, usage

    # ------------------------------------------------------------------ proposal

    def _document(self, file_name: str) -> dict[str, object] | None:
        return query_one(self.conn, "SELECT * FROM kb_documents WHERE file_name = ?", [file_name])

    def _checks(
        self, question: str, queries: list[str], draft: ProposalDraftOutput, doc: dict[str, object]
    ) -> tuple[list[VerificationCheck], Usage]:
        known = {
            str(r["file_name"]) for r in query(self.conn, "SELECT file_name FROM kb_documents")
        }
        checks: list[VerificationCheck] = []

        unknown = [s for s in draft.sources_used if s.endswith(".md") and s not in known]
        checks.append(
            VerificationCheck(
                name="SOURCES_EXIST",
                passed=bool(draft.sources_used) and not unknown,
                severity="HIGH",
                detail=f"Unknown sources: {unknown}"
                if unknown
                else f"Sources: {draft.sources_used}",
            )
        )
        headings = {
            h.strip().lower() for h in re.findall(r"^## (.+)$", str(doc["content"]), re.MULTILINE)
        }
        duplicate = draft.proposed_title.strip().lower() in headings
        checks.append(
            VerificationCheck(
                name="NOT_ALREADY_IN_DOCUMENT",
                passed=not duplicate,
                severity="MEDIUM",
                detail="A section with this title already exists" if duplicate else "New section",
            )
        )
        placeholders = re.findall(r"\[[^\]]+\]", draft.proposed_content)
        checks.append(
            VerificationCheck(
                name="NO_PLACEHOLDER_LEFT",
                passed=not placeholders,
                severity="MEDIUM",
                detail=f"Reviewer must fill: {placeholders}" if placeholders else "None left",
            )
        )

        # The decisive test: does the updated document now answer the original question?
        updated = dict(doc)
        updated["content"] = append_section(
            str(doc["content"]), draft.proposed_title, draft.proposed_content
        )
        others = [
            d
            for d in query(self.conn, "SELECT * FROM kb_documents")
            if d["file_name"] != doc["file_name"]
        ]
        tester = KnowledgeAgent(
            self.llm, KnowledgeIndex.from_documents([*others, updated]), self.top_k
        )
        hits = tester.retrieve(question, queries, str(doc["file_name"]))
        usage = Usage()
        try:
            answer, usage = tester.answer(question, hits)
            closes = not answer.insufficient_evidence
            detail = "Updated document answers the question" if closes else "Still insufficient"
        except LLMError as e:
            closes, detail = False, f"Could not test: {e}"
        checks.append(
            VerificationCheck(name="CLOSES_THE_GAP", passed=closes, severity="HIGH", detail=detail)
        )
        return checks, usage

    def propose(self, gap: GapCandidate) -> tuple[ProposalResult | None, Usage]:
        if gap.target_document is None:
            return None, Usage()
        doc = self._document(gap.target_document)
        if doc is None:
            return None, Usage()
        samples = query(
            self.conn,
            "SELECT ticket_ref, description FROM tickets WHERE ticket_id IN "  # noqa: S608 - only ? placeholders are interpolated
            f"({', '.join('?' for _ in gap.ticket_ids)}) ORDER BY created_at LIMIT 5",
            gap.ticket_ids,
        )
        evidence = DataInvestigationAgent(self.conn).run(gap.representative_ticket_id)
        related = [p for p in gap.knowledge.passages if p.file_name != gap.target_document]
        related_docs = [self._document(p.file_name) for p in related]
        user = "\n\n".join(
            [
                f"TARGET DOCUMENT ({doc['file_name']}, version {doc['version']}):\n{doc['content']}",
                "OTHER RELEVANT DOCUMENTS:\n"
                + "\n\n".join(
                    f"[{d['file_name']}]\n{d['content']}" for d in related_docs if d is not None
                ),
                "OPEN TICKETS ASKING THE SAME QUESTION:\n"
                + "\n".join(f"- {s['ticket_ref']}: {s['description']}" for s in samples),
                "WHAT THE KNOWLEDGE AGENT FOUND:\n"
                + (gap.knowledge.answer.answer if gap.knowledge.answer else "nothing"),
                "DATA FOR THE FIRST TICKET:\n"
                + _compact(evidence.model_dump(include={"inland", "price_lines", "quotes"})),
            ]
        )
        try:
            result = self.llm.structured(
                task="maintenance.propose",
                system=PROPOSAL_SYSTEM,
                user=user,
                schema=ProposalDraftOutput,
            )
        except LLMError:
            return None, Usage()
        draft = result.output
        checks, check_usage = self._checks(gap.question, gap.knowledge.queries, draft, doc)
        with transaction(self.conn):
            cur = self.conn.execute(
                "INSERT INTO kb_proposals (created_at, subcategory, related_ticket_ids, "
                "target_document, question, search_queries, identified_gap, proposed_title, "
                "proposed_content, rationale, sources, checks) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    utcnow(),
                    gap.subcategory,
                    json.dumps(gap.ticket_ids),
                    gap.target_document,
                    gap.question,
                    json.dumps(gap.knowledge.queries),
                    gap.knowledge.answer.answer if gap.knowledge.answer else "",
                    draft.proposed_title,
                    draft.proposed_content,
                    draft.rationale,
                    json.dumps(draft.sources_used),
                    json.dumps([c.model_dump() for c in checks]),
                ),
            )
        return (
            ProposalResult(
                proposal_id=int(cur.lastrowid or 0),
                subcategory=gap.subcategory,
                target_document=gap.target_document,
                proposed_title=draft.proposed_title,
                checks=checks,
            ),
            result.usage + check_usage,
        )

    def revise(
        self, proposal_id: int, content: str, title: str | None = None
    ) -> tuple[ProposalResult, Usage]:
        """Apply a reviewer's edit (e.g. filled placeholders) and re-run the pre-publication checks."""
        p = query_one(self.conn, "SELECT * FROM kb_proposals WHERE proposal_id = ?", [proposal_id])
        if p is None:
            raise ReviewError(f"proposal {proposal_id} not found")
        if p["status"] != "PENDING":
            raise ReviewError(
                f"proposal {proposal_id} is {p['status']}, only PENDING can be revised"
            )
        doc = self._document(str(p["target_document"]))
        if doc is None:
            raise ReviewError(f"target document {p['target_document']} no longer exists")
        draft = ProposalDraftOutput(
            proposed_title=title or str(p["proposed_title"]),
            proposed_content=content,
            rationale=str(p["rationale"]),
            sources_used=json.loads(str(p["sources"])),
        )
        checks, usage = self._checks(
            str(p["question"]), json.loads(str(p["search_queries"])), draft, doc
        )
        with transaction(self.conn):
            self.conn.execute(
                "UPDATE kb_proposals SET proposed_title = ?, proposed_content = ?, checks = ?, "
                "revised_at = ? WHERE proposal_id = ?",
                (
                    draft.proposed_title,
                    draft.proposed_content,
                    json.dumps([c.model_dump() for c in checks]),
                    utcnow(),
                    proposal_id,
                ),
            )
        return (
            ProposalResult(
                proposal_id=proposal_id,
                subcategory=str(p["subcategory"]),
                target_document=str(p["target_document"]),
                proposed_title=draft.proposed_title,
                checks=checks,
            ),
            usage,
        )

    def run(self) -> MaintenanceReport:
        gaps, covered, skipped, usage = self.detect_gaps()
        proposals = []
        for gap in gaps:
            proposal, u = self.propose(gap)
            usage = usage + u
            if proposal:
                proposals.append(proposal)
        return MaintenanceReport(len(gaps) + len(covered), covered, skipped, proposals, usage)


# ---------------------------------------------------------------------- human review


def review_proposal(
    conn: sqlite3.Connection, proposal_id: int, decision: str, reviewer: str, comment: str = ""
) -> dict[str, object]:
    """Approve or reject a pending proposal. Approval versions and publishes the document."""
    if decision not in {"APPROVED", "REJECTED"}:
        raise ReviewError("decision must be APPROVED or REJECTED")
    if not reviewer.strip():
        raise ReviewError("a reviewer name is required")
    p = query_one(conn, "SELECT * FROM kb_proposals WHERE proposal_id = ?", [proposal_id])
    if p is None:
        raise ReviewError(f"proposal {proposal_id} not found")
    if p["status"] != "PENDING":
        raise ReviewError(f"proposal {proposal_id} is {p['status']}, only PENDING can be reviewed")
    checks = [VerificationCheck.model_validate(c) for c in json.loads(str(p["checks"]))]
    failing = [c.name for c in checks if not c.passed]
    if decision == "APPROVED" and failing:
        raise ReviewError(f"cannot approve: failing pre-publication checks {failing}")

    now = utcnow()
    with transaction(conn):
        conn.execute(
            "UPDATE kb_proposals SET status = ?, reviewed_by = ?, reviewed_at = ?, "
            "review_comment = ? WHERE proposal_id = ?",
            (decision, reviewer, now, comment, proposal_id),
        )
        if decision == "REJECTED":
            return {"proposal_id": proposal_id, "status": "REJECTED"}

        file_name = str(p["target_document"])
        doc = query_one(conn, "SELECT * FROM kb_documents WHERE file_name = ?", [file_name])
        if doc is None:
            raise ReviewError(f"target document {file_name} no longer exists")
        last = query_one(
            conn,
            "SELECT MAX(version_number) AS v FROM kb_versions WHERE file_name = ?",
            [file_name],
        )
        number = int(last["v"]) if last and last["v"] is not None else 0
        if number == 0:  # keep the original as version 1
            conn.execute(
                "INSERT INTO kb_versions (file_name, version_number, content, change_note, "
                "created_by, created_at, is_current) VALUES (?, 1, ?, ?, 'SYSTEM', ?, 0)",
                (file_name, doc["content"], f"Original document (v{doc['version']})", now),
            )
            number = 1
        conn.execute("UPDATE kb_versions SET is_current = 0 WHERE file_name = ?", (file_name,))
        content = append_section(
            str(doc["content"]), str(p["proposed_title"]), str(p["proposed_content"])
        )
        conn.execute(
            "INSERT INTO kb_versions (file_name, version_number, content, change_note, proposal_id, "
            "created_by, created_at, is_current) VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
            (
                file_name,
                number + 1,
                content,
                f"Added section: {p['proposed_title']}",
                proposal_id,
                reviewer,
                now,
            ),
        )
        new_version = bump_version(str(doc["version"]))
        conn.execute(
            "UPDATE kb_documents SET content = ?, version = ? WHERE file_name = ?",
            (content, new_version, file_name),
        )
    return {
        "proposal_id": proposal_id,
        "status": "APPROVED",
        "file_name": file_name,
        "document_version": new_version,
        "history_version": number + 1,
    }
