"""Typed contracts between agents.

LLM outputs are validated against the ``*Output`` models (structured outputs); the other models
are what agents return to the orchestrator and what the UI and the evaluation read.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Decision(StrEnum):
    ANSWER = "ANSWER"
    ASK_CLARIFICATION = "ASK_CLARIFICATION"
    ESCALATE = "ESCALATE"
    DETECT_KB_GAP = "DETECT_KB_GAP"


Confidence = Literal["HIGH", "MEDIUM", "LOW"]


# ---------------------------------------------------------------- LLM output schemas


class SearchPlanOutput(BaseModel):
    queries: list[str] = Field(description="One to three short keyword search queries.")


class KnowledgeAnswerOutput(BaseModel):
    answer: str
    sources_used: list[str] = Field(description="File names of the documents actually used.")
    confidence: Confidence
    insufficient_evidence: bool


class ResolutionOutput(BaseModel):
    decision: Literal["ANSWER", "ASK_CLARIFICATION", "ESCALATE"]
    response: str
    evidence_used: list[str]
    sources: list[str] = Field(description="Document file names and data references (QS-, BK-…).")
    missing_information: list[str]
    confidence: Confidence


class ProposalDraftOutput(BaseModel):
    proposed_title: str
    proposed_content: str
    rationale: str
    sources_used: list[str]


# ---------------------------------------------------------------- agent results


class PassageRef(BaseModel):
    file_name: str
    section: str
    version: str
    score: float


class KnowledgeResult(BaseModel):
    question: str
    queries: list[str] = Field(default_factory=list)
    passages: list[PassageRef] = Field(default_factory=list)
    answer: KnowledgeAnswerOutput | None = None
    error: str | None = None

    @property
    def insufficient(self) -> bool:
        """True when the knowledge base gave no usable answer, including on failure."""
        return self.answer is None or self.answer.insufficient_evidence

    @property
    def retrieved_files(self) -> set[str]:
        return {p.file_name for p in self.passages}


class DataEvidence(BaseModel):
    """Everything the data investigation agent found, as plain JSON-able records."""

    ticket: dict[str, Any]
    references: dict[str, list[str]] = Field(default_factory=dict)
    quotes: list[dict[str, Any]] = Field(default_factory=list)
    bookings: list[dict[str, Any]] = Field(default_factory=list)
    price_lines: list[dict[str, Any]] = Field(default_factory=list)
    expired_lines: list[dict[str, Any]] = Field(default_factory=list)
    current_lines: list[dict[str, Any]] = Field(default_factory=list)
    overlapping_lines: list[dict[str, Any]] = Field(default_factory=list)
    inland: dict[str, Any] | None = None
    similar_open_tickets: list[dict[str, Any]] = Field(default_factory=list)
    reporter: dict[str, Any] | None = None

    @property
    def has_records(self) -> bool:
        return bool(self.quotes or self.bookings or self.price_lines)

    @property
    def draft_lines(self) -> list[dict[str, Any]]:
        return [line for line in self.price_lines if line.get("status") == "DRAFT"]


class PatternResult(BaseModel):
    open_tickets_in_subcategory: int
    threshold: int
    is_recurring: bool
    kb_gap: bool


class VerificationCheck(BaseModel):
    name: str
    passed: bool
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    detail: str
    overrides: bool = False


class VerificationResult(BaseModel):
    checks: list[VerificationCheck]
    proposed_decision: Decision
    verified_decision: Decision
    override_reason: str | None = None

    @property
    def overridden(self) -> bool:
        return self.verified_decision != self.proposed_decision

    @property
    def passed_count(self) -> int:
        return sum(c.passed for c in self.checks)


class Resolution(BaseModel):
    """The final answer sent back to the requester."""

    decision: Decision
    response: str
    evidence_used: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    confidence: Confidence = "LOW"


class StepTrace(BaseModel):
    agent: str
    action: str
    status: Literal["OK", "ERROR", "SKIPPED"]
    duration_ms: int
    summary: str = ""
    cost_usd: float = 0.0


class TicketRun(BaseModel):
    execution_id: str
    ticket_id: int
    ticket_ref: str
    category: str
    subcategory: str
    knowledge: KnowledgeResult | None = None
    data: DataEvidence | None = None
    pattern: PatternResult | None = None
    proposed: Resolution | None = None
    verification: VerificationResult | None = None
    final: Resolution
    steps: list[StepTrace] = Field(default_factory=list)
    duration_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
