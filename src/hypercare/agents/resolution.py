"""Resolution agent: decide and write the reply from the evidence gathered by the other agents."""

from __future__ import annotations

import json

from hypercare.agents.schemas import (
    DataEvidence,
    Decision,
    KnowledgeResult,
    Resolution,
    ResolutionOutput,
)
from hypercare.llm.base import LLMClient, LLMError, Usage

RESOLUTION_SYSTEM = """\
You are the resolution agent of the hypercare support team for TideRate (pricing platform) and
QuotePort (quoting portal) at Northwind Shipping. You receive a support ticket, the answer of the
knowledge agent (with its sources) and the records found by the data investigation agent.

Decide one of:
- ANSWER: the evidence supports a clear, complete answer.
- ASK_CLARIFICATION: the ticket lacks information you need (for example a quote or route reference).
- ESCALATE: the evidence is missing or contradictory, or the documented process says a human must
  handle it.

Rules from the documentation that you must apply:
- Never invent rates, dates, contacts or procedures; use only the evidence provided.
- A DRAFT price line cannot be confirmed, quoted or booked: escalate to the Pricing Manager.
- Overlapping ACTIVE lines for the same product: do not choose one, escalate to the Pricing desk.
- An EXPIRED line or quote cannot be used for new business; say so and point to the current line.
- For a price difference, explain the root cause with the actual amounts (base rate, each surcharge,
  all-in price, validity dates).
- Discounts and special rates are not handled by hypercare: escalate.

The response is sent to the sales or customer service colleague who opened the ticket: be specific
and brief (under 150 words). List in sources the document file names and the data references you
used (quote, booking or price line references)."""


def _compact(obj: object) -> str:
    return json.dumps(obj, default=str, separators=(",", ":"), sort_keys=True)


def build_prompt(
    description: str, knowledge: KnowledgeResult | None, data: DataEvidence | None
) -> str:
    parts = [f"TICKET:\n{description}"]
    if knowledge is None:
        parts.append("KNOWLEDGE AGENT: not run.")
    elif knowledge.answer is None:
        parts.append(f"KNOWLEDGE AGENT: failed ({knowledge.error}).")
    else:
        parts.append("KNOWLEDGE AGENT:\n" + _compact(knowledge.answer.model_dump()))
    if data is None:
        parts.append("DATA INVESTIGATION: not run.")
    else:
        parts.append(
            "DATA INVESTIGATION:\n"
            + _compact(data.model_dump(exclude={"similar_open_tickets", "reporter"}))
        )
    return "\n\n".join(parts)


def escalation(reason: str) -> Resolution:
    return Resolution(
        decision=Decision.ESCALATE,
        response=f"Automatic resolution was not possible ({reason}). The ticket is escalated to a "
        "hypercare agent.",
        missing_information=[reason],
        confidence="LOW",
    )


class ResolutionAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self, description: str, knowledge: KnowledgeResult | None, data: DataEvidence | None
    ) -> tuple[Resolution, Usage]:
        try:
            result = self.llm.structured(
                task="resolution.decide",
                system=RESOLUTION_SYSTEM,
                user=build_prompt(description, knowledge, data),
                schema=ResolutionOutput,
            )
        except LLMError as e:
            return escalation(str(e)), Usage()
        out = result.output
        return (
            Resolution(
                decision=Decision(out.decision),
                response=out.response,
                evidence_used=out.evidence_used,
                sources=out.sources,
                missing_information=out.missing_information,
                confidence=out.confidence,
            ),
            result.usage,
        )
