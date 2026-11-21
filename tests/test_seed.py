from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hypercare.db import query, query_one
from hypercare.seed import load_kb_documents, parse_kb_document, ticket_ref
from tests.conftest import KB_DIR


def test_counts(conn: sqlite3.Connection) -> None:
    assert query_one(conn, "SELECT COUNT(*) AS n FROM tickets")["n"] == 30  # type: ignore[index]
    assert query_one(conn, "SELECT COUNT(*) AS n FROM kb_documents")["n"] == 11  # type: ignore[index]


def test_ticket_refs() -> None:
    assert ticket_ref(7) == "HC-2026-0007"


def test_every_quote_matches_its_price_line(conn: sqlite3.Connection) -> None:
    """Data integrity: a quote is the all-in price of its line for the quoted equipment."""
    rows = query(
        conn,
        "SELECT q.quote_ref, q.equipment_size, q.quoted_rate, pls.charges_included, "
        "pl.rate_20st, pl.rate_40st, pl.rate_40hc, pl.priceline_id "
        "FROM quotes q JOIN price_lines pl USING (priceline_id) JOIN price_lists pls USING (pricelist_id)",
    )
    for r in rows:
        size = r["equipment_size"].lower()
        surcharges = query_one(
            conn,
            f"SELECT COALESCE(SUM(amount_{size}), 0) AS s FROM price_line_surcharges WHERE priceline_id = ?",  # noqa: S608
            [r["priceline_id"]],
        )
        assert surcharges is not None
        expected = r[f"rate_{size}"] + (0 if r["charges_included"] else surcharges["s"])
        assert r["quoted_rate"] == expected, r["quote_ref"]


def test_kb_documents_have_front_matter() -> None:
    docs = load_kb_documents(KB_DIR)
    assert {d.source_type for d in docs} == {"DOC", "VIDEO_TRANSCRIPT"}
    assert all(d.content and not d.content.startswith("---") for d in docs)


def test_front_matter_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.md"
    bad.write_text("# no front matter")
    with pytest.raises(ValueError, match="missing front matter"):
        parse_kb_document(bad)
    bad.write_text("---\ntitle: x\n---\nbody")
    with pytest.raises(ValueError, match="missing front matter keys"):
        parse_kb_document(bad)
