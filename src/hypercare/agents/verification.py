"""Verification agent: deterministic checks on the proposed reply before it is sent.

The checks encode the documented business rules and grounding requirements. A failing check marked
``overrides`` turns the decision into ESCALATE. The orchestrator fails closed: if verification
itself cannot run, the ticket is escalated.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from hypercare.agents.schemas import (
    DataEvidence,
    Decision,
    KnowledgeResult,
    Resolution,
    VerificationCheck,
    VerificationResult,
)

_NUMBER = re.compile(r"\d[\d,  ]*\d|\d")
_EXPIRY_WORDS = (
    "expired",
    "expire",
    "previous",
    "former",
    "old ",
    "older",
    "no longer",
    "outdated",
    "last month",
    "earlier",
    "replaced",
)
_DATA_REF = re.compile(r"^(QS-|BK-|PL-|HC-)|line|table|price_lines|quotes|bookings", re.IGNORECASE)
SIZES = ("20st", "40st", "40hc")


def numbers_in(text: str) -> set[int]:
    """Integers mentioned in a text, accepting thousands separators (1,850 / 1 850 / 1850)."""
    out = set()
    for match in _NUMBER.findall(text):
        digits = re.sub(r"[,  ]", "", match)
        if digits.isdigit():
            out.add(int(digits))
    return out


def _amounts(lines: Iterable[dict[str, Any]]) -> set[int]:
    values: set[int] = set()
    for line in lines:
        for size in SIZES:
            for key in (f"rate_{size}", f"all_in_{size}"):
                if line.get(key) is not None:
                    values.add(int(line[key]))
    return values


def _na(name: str) -> VerificationCheck:
    return VerificationCheck(name=name, passed=True, severity="LOW", detail="Not applicable")


class VerificationAgent:
    def run(
        self,
        proposed: Resolution,
        knowledge: KnowledgeResult | None,
        data: DataEvidence | None,
    ) -> VerificationResult:
        answering = proposed.decision == Decision.ANSWER
        response = proposed.response
        mentioned = numbers_in(response)
        lowered = response.lower()
        checks: list[VerificationCheck] = []

        # 1. The knowledge base said it could not answer, and no data backs the answer either.
        name = "KB_INSUFFICIENT_BUT_ANSWERED"
        if answering and knowledge is not None and knowledge.insufficient:
            backed = data is not None and data.has_records
            checks.append(
                VerificationCheck(
                    name=name,
                    passed=backed,
                    severity="LOW" if backed else "HIGH",
                    detail="Documentation was insufficient but the answer relies on data records"
                    if backed
                    else "Answered although the knowledge base flagged insufficient evidence and "
                    "no data record supports the answer",
                    overrides=not backed,
                )
            )
        else:
            checks.append(_na(name))

        # 2. A DRAFT line can never be confirmed, quoted or booked.
        name = "DRAFT_LINE_ANSWERED"
        drafts = data.draft_lines if data else []
        if answering and drafts:
            refs = ", ".join(f"{d['pricelist_code']} line {d['line_ref']}" for d in drafts)
            checks.append(
                VerificationCheck(
                    name=name,
                    passed=False,
                    severity="CRITICAL",
                    detail=f"Answer given on DRAFT (unpublished) line(s) {refs}; must be escalated",
                    overrides=True,
                )
            )
        else:
            checks.append(_na(name))

        # 3. An expired rate must not be presented as current.
        name = "EXPIRED_RATE_AS_CURRENT"
        if answering and data:
            expired = [ln for ln in data.price_lines if ln.get("status") == "EXPIRED"]
            expired += data.expired_lines
            used = mentioned & _amounts(expired)
            flagged = any(word in lowered for word in _EXPIRY_WORDS)
            ok = not used or flagged
            checks.append(
                VerificationCheck(
                    name=name,
                    passed=ok,
                    severity="LOW" if ok else "HIGH",
                    detail="No expired rate presented as current"
                    if ok
                    else f"Mentions expired rate(s) {sorted(used)} without saying they expired",
                    overrides=not ok,
                )
            )
        else:
            checks.append(_na(name))

        # 4. An answer must cite sources, and cited documents must have been retrieved.
        name = "UNSUPPORTED_SOURCES"
        if answering:
            retrieved = knowledge.retrieved_files if knowledge else set()
            docs = [s for s in proposed.sources if s.endswith(".md")]
            unknown = [s for s in docs if s not in retrieved]
            others = [s for s in proposed.sources if not s.endswith(".md")]
            data_cited = any(_DATA_REF.search(s) for s in others)
            ok = not unknown and (bool(docs) or data_cited)
            if unknown:
                detail = f"Cites documents that were not retrieved: {unknown}"
            elif not ok:
                detail = "Answer without any document or data source"
            else:
                detail = f"{len(docs)} document(s) and {len(others)} data reference(s) cited"
            checks.append(
                VerificationCheck(
                    name=name,
                    passed=ok,
                    severity="LOW" if ok else "HIGH",
                    detail=detail,
                    overrides=not ok,
                )
            )
        else:
            checks.append(_na(name))

        # 5. A pricing answer should state the actual amounts it is about.
        name = "RATE_NOT_IN_RESPONSE"
        if answering and data and (data.price_lines or data.quotes):
            expected = _amounts(data.price_lines)
            expected |= {int(q["quoted_rate"]) for q in data.quotes}
            expected |= {int(b["total_amount"]) for b in data.bookings}
            ok = bool(mentioned & expected)
            checks.append(
                VerificationCheck(
                    name=name,
                    passed=ok,
                    severity="LOW" if ok else "MEDIUM",
                    detail="The answer states the amounts from the data"
                    if ok
                    else "No amount from the referenced lines, quotes or bookings in the answer",
                )
            )
        else:
            checks.append(_na(name))

        # 6. Overlapping ACTIVE lines: hypercare must not choose between them.
        name = "OVERLAPPING_LINES_ANSWERED"
        overlaps = data.overlapping_lines if data else []
        if answering and overlaps:
            refs = ", ".join(f"{o['pricelist_code']} line {o['line_ref']}" for o in overlaps)
            checks.append(
                VerificationCheck(
                    name=name,
                    passed=False,
                    severity="CRITICAL",
                    detail=f"Answered while ACTIVE lines overlap ({refs}); must be escalated",
                    overrides=True,
                )
            )
        else:
            checks.append(_na(name))

        blocking = [c for c in checks if not c.passed and c.overrides]
        return VerificationResult(
            checks=checks,
            proposed_decision=proposed.decision,
            verified_decision=Decision.ESCALATE if blocking else proposed.decision,
            override_reason=blocking[0].detail if blocking else None,
        )


def failed_closed(proposed: Resolution, error: str) -> VerificationResult:
    """Result used when verification could not run: the ticket is escalated."""
    return VerificationResult(
        checks=[
            VerificationCheck(
                name="VERIFICATION_ERROR",
                passed=False,
                severity="CRITICAL",
                detail=f"Verification could not run ({error}); escalating",
                overrides=True,
            )
        ],
        proposed_decision=proposed.decision,
        verified_decision=Decision.ESCALATE,
        override_reason=f"Verification could not run ({error})",
    )
