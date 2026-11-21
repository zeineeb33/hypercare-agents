from __future__ import annotations

import sqlite3

import pytest

from hypercare.agents.data_investigation import (
    DataInvestigationAgent,
    extract_references,
    has_references,
)


def test_extract_references() -> None:
    refs = extract_references(
        "See qs-10231, BK-55012 and PL-2026-041 line 12; also PL-2026-052, line 7"
    )
    assert refs == {
        "quotes": ["QS-10231"],
        "bookings": ["BK-55012"],
        "price_lines": ["PL-2026-041 line 12", "PL-2026-052 line 7"],
    }
    assert not has_references("The price is wrong")


@pytest.fixture
def agent(conn: sqlite3.Connection) -> DataInvestigationAgent:
    return DataInvestigationAgent(conn)


def test_all_in_price_explains_the_gap(agent: DataInvestigationAgent) -> None:
    e = agent.run(3)
    line = e.price_lines[0]
    assert (line["rate_40hc"], line["all_in_40hc"]) == (1850, 2150)
    assert {s["charge_code"] for s in line["surcharges"]} == {"BAF", "LSS"}
    assert e.quotes[0]["quoted_rate"] == 2150
    assert e.overlapping_lines == []  # the SOC line on the same lane is another product


def test_expired_line_replaced(agent: DataInvestigationAgent) -> None:
    e = agent.run(4)
    statuses = {ln["line_ref"]: ln["status"] for ln in e.price_lines}
    assert statuses == {3: "EXPIRED", 5: "ACTIVE"}
    assert next(q for q in e.quotes if q["quote_ref"] == "QS-10198")["expired_on_ticket_date"]


def test_draft_line(agent: DataInvestigationAgent) -> None:
    assert [d["line_ref"] for d in agent.run(6).draft_lines] == [7]
    assert agent.run(7).draft_lines  # found through the quote


def test_overlapping_lines(agent: DataInvestigationAgent) -> None:
    assert sorted(ln["line_ref"] for ln in agent.run(8).overlapping_lines) == [2, 3]


def test_missing_inland_rate(agent: DataInvestigationAgent) -> None:
    e = agent.run(9)
    assert e.inland is not None
    assert e.inland["inland_point"] == "FRLYS"
    assert e.inland["pre_carriage_rate_found"] is False
    assert len(e.similar_open_tickets) == 4  # the resolved Lille ticket is not counted


def test_booking_found_through_reference(agent: DataInvestigationAgent) -> None:
    e = agent.run(16)
    assert e.bookings[0]["total_amount"] == 4300
    assert e.has_records


def test_unknown_ticket(agent: DataInvestigationAgent) -> None:
    with pytest.raises(LookupError):
        agent.run(999)
