"""Orchestrator: route a ticket through the agents, trace every step, return the final decision."""

from __future__ import annotations

import sqlite3
import time

from hypercare.agents.data_investigation import DataInvestigationAgent, has_references
from hypercare.agents.knowledge import KnowledgeAgent
from hypercare.agents.resolution import ResolutionAgent, escalation
from hypercare.agents.schemas import (
    DataEvidence,
    Decision,
    KnowledgeResult,
    PatternResult,
    Resolution,
    TicketRun,
    VerificationResult,
)
from hypercare.agents.verification import VerificationAgent, failed_closed
from hypercare.db import query_one
from hypercare.llm.base import LLMClient
from hypercare.retrieval import KnowledgeIndex
from hypercare.tracing import Tracer, new_execution_id

DATA_CATEGORIES = {"PRICING_ISSUE", "TECHNICAL"}

CLARIFICATION_QUESTIONS = [
    "The quote (QS-...) or booking (BK-...) reference",
    "The route: port or inland point of loading and port of discharge",
    "The equipment type and the date or period concerned",
]


def count_open_tickets(conn: sqlite3.Connection, subcategory: str) -> int:
    row = query_one(
        conn,
        "SELECT COUNT(*) AS n FROM tickets WHERE subcategory = ? AND status != 'RESOLVED'",
        [subcategory],
    )
    return int(row["n"]) if row else 0


class Orchestrator:
    def __init__(
        self,
        conn: sqlite3.Connection,
        llm: LLMClient,
        index: KnowledgeIndex,
        *,
        top_k: int = 4,
        recurring_min_tickets: int = 4,
    ) -> None:
        self.conn = conn
        self.knowledge_agent = KnowledgeAgent(llm, index, top_k)
        self.data_agent = DataInvestigationAgent(conn)
        self.resolution_agent = ResolutionAgent(llm)
        self.verification_agent = VerificationAgent()
        self.recurring_min_tickets = recurring_min_tickets

    def run(self, ticket_id: int) -> TicketRun:
        start = time.perf_counter()
        ticket = query_one(self.conn, "SELECT * FROM tickets WHERE ticket_id = ?", [ticket_id])
        if ticket is None:
            raise LookupError(f"ticket {ticket_id} not found")
        run_id = new_execution_id()
        tracer = Tracer(self.conn, run_id, ticket_id)
        description = str(ticket["description"])
        category, subcategory = str(ticket["category"]), str(ticket["subcategory"])

        knowledge: KnowledgeResult | None = None
        data: DataEvidence | None = None
        pattern: PatternResult | None = None
        proposed: Resolution
        verification: VerificationResult | None = None

        # 1. Classification: which agents does this ticket need?
        with tracer.step("ORCHESTRATOR", "CLASSIFY") as s:
            referenced = bool(
                ticket["priceline_id"] or ticket["route_id"] or has_references(description)
            )
            needs_clarification = category == "PRICING_ISSUE" and not referenced
            needs_data = referenced or category in DATA_CATEGORIES
            s.input_summary = f"{ticket['ticket_ref']} {category}/{subcategory}"
            s.summary = f"references={referenced}, data={needs_data}, clarify={needs_clarification}"

        if needs_clarification:
            # A pricing ticket with nothing to investigate: ask instead of guessing.
            with tracer.step("RESOLUTION_AGENT", "ASK_CLARIFICATION") as s:
                proposed = Resolution(
                    decision=Decision.ASK_CLARIFICATION,
                    response="This ticket has no reference we can investigate (no quote, "
                    "booking, route or price line). To check it without guessing, please send: "
                    + "; ".join(CLARIFICATION_QUESTIONS)
                    + ".",
                    missing_information=CLARIFICATION_QUESTIONS,
                    confidence="HIGH",
                )
                s.decision = proposed.decision
                s.summary = "No usable reference in a pricing ticket"
        else:
            # 2. Knowledge agent.
            with tracer.step("KNOWLEDGE_AGENT", "SEARCH_AND_ANSWER") as s:
                knowledge, usage = self.knowledge_agent.run(description, ticket["related_doc"])
                s.add_usage(usage)
                s.input_summary = "; ".join(knowledge.queries)
                s.sources = sorted(knowledge.retrieved_files)
                s.summary = (
                    f"error: {knowledge.error}"
                    if knowledge.error
                    else f"insufficient={knowledge.insufficient}, "
                    f"{len(knowledge.passages)} passage(s)"
                )

            # 3. Data investigation agent.
            with tracer.step("DATA_INVESTIGATION_AGENT", "INVESTIGATE") as s:
                if needs_data:
                    data = self.data_agent.run(ticket_id)
                    s.summary = (
                        f"{len(data.quotes)} quote(s), {len(data.price_lines)} line(s), "
                        f"{len(data.draft_lines)} draft, {len(data.overlapping_lines)} overlapping"
                    )
                else:
                    s.skipped = True
                    s.summary = "Documentation question: no data needed"

            # 4. Pattern detection: a recurring issue that the documentation cannot answer.
            with tracer.step("ORCHESTRATOR", "PATTERN_DETECTION") as s:
                n_open = count_open_tickets(self.conn, subcategory)
                recurring = n_open >= self.recurring_min_tickets
                pattern = PatternResult(
                    open_tickets_in_subcategory=n_open,
                    threshold=self.recurring_min_tickets,
                    is_recurring=recurring,
                    kb_gap=recurring and knowledge.answer is not None and knowledge.insufficient,
                )
                s.summary = f"{n_open} open ticket(s) in {subcategory}, kb_gap={pattern.kb_gap}"

            # 5. Resolution agent (skipped for a KB gap: the answer is the gap itself).
            with tracer.step("RESOLUTION_AGENT", "DECIDE") as s:
                if pattern.kb_gap and knowledge.answer is not None:
                    proposed = Resolution(
                        decision=Decision.DETECT_KB_GAP,
                        response=f"Recurring issue: {n_open} open tickets in {subcategory} "
                        "that the knowledge base does not cover. What the documentation says: "
                        f"{knowledge.answer.answer} A knowledge-base update has been flagged "
                        "for review; meanwhile the ticket goes to the pricing desk.",
                        evidence_used=[f"{n_open} open tickets in {subcategory}"],
                        sources=sorted(knowledge.retrieved_files),
                        confidence="HIGH",
                    )
                    s.summary = "KB gap: resolution skipped"
                else:
                    proposed, usage = self.resolution_agent.run(description, knowledge, data)
                    s.add_usage(usage)
                    s.summary = proposed.response[:200]
                s.decision = proposed.decision
                s.sources = proposed.sources

        # 6. Verification agent (fails closed).
        with tracer.step("VERIFICATION_AGENT", "VERIFY") as s:
            try:
                verification = self.verification_agent.run(proposed, knowledge, data)
            except Exception as e:
                verification = failed_closed(proposed, f"{type(e).__name__}: {e}")
            s.decision = verification.verified_decision
            s.summary = (
                f"{verification.passed_count}/{len(verification.checks)} checks passed, "
                f"overridden={verification.overridden}"
            )

        final = proposed
        if verification.overridden:
            reason = verification.override_reason or "verification failed"
            final = escalation(f"verification: {reason}").model_copy(
                update={"sources": proposed.sources, "evidence_used": proposed.evidence_used}
            )

        with tracer.step("ORCHESTRATOR", "COMPLETE") as s:
            s.decision = final.decision
            s.summary = f"final={final.decision}"

        usage = tracer.usage
        return TicketRun(
            execution_id=run_id,
            ticket_id=ticket_id,
            ticket_ref=str(ticket["ticket_ref"]),
            category=category,
            subcategory=subcategory,
            knowledge=knowledge,
            data=data,
            pattern=pattern,
            proposed=proposed,
            verification=verification,
            final=final,
            steps=tracer.steps,
            duration_ms=int((time.perf_counter() - start) * 1000),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=usage.cost_usd,
        )
