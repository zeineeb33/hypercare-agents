"""Data investigation agent: deterministic, parameterised SQL over the pricing data.

It follows the references found in the ticket (quotes, bookings, price lines), then looks for the
usual root causes: surcharges on top of a base rate, an expired line replaced by a new one, DRAFT
lines, overlapping ACTIVE lines and missing inland rates.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from hypercare.agents.schemas import DataEvidence
from hypercare.db import Row, query, query_one

QUOTE_REF = re.compile(r"\bQS-\d{4,6}\b", re.IGNORECASE)
BOOKING_REF = re.compile(r"\bBK-\d{4,6}\b", re.IGNORECASE)
LINE_REF = re.compile(r"\b(PL-\d{4}-\d{3})\b[\s,]*line\s*(\d+)", re.IGNORECASE)

SIZES = ("20st", "40st", "40hc")

_LINE_SELECT = """
SELECT pl.priceline_id, pls.pricelist_code, pl.line_ref, pl.status, pl.valid_from, pl.valid_to,
       pl.currency, pl.rate_20st, pl.rate_40st, pl.rate_40hc, pl.has_soc, pl.has_haz,
       pls.charges_included, r.route_id, r.pol_code, r.pol_name, r.pod_code, r.pod_name,
       r.via_port, r.is_inland, c.commodity_id, c.commodity_code, c.commodity_name
FROM price_lines pl
JOIN price_lists pls ON pls.pricelist_id = pl.pricelist_id
JOIN routes r ON r.route_id = pl.route_id
JOIN commodities c ON c.commodity_id = pl.commodity_id
"""


def extract_references(text: str) -> dict[str, list[str]]:
    return {
        "quotes": sorted({m.upper() for m in QUOTE_REF.findall(text)}),
        "bookings": sorted({m.upper() for m in BOOKING_REF.findall(text)}),
        "price_lines": sorted({f"{pl.upper()} line {n}" for pl, n in LINE_REF.findall(text)}),
    }


def has_references(text: str) -> bool:
    return any(extract_references(text).values())


def _placeholders(values: list[Any]) -> str:
    return ", ".join("?" for _ in values)


class DataInvestigationAgent:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def _line_details(self, priceline_ids: list[int]) -> list[Row]:
        if not priceline_ids:
            return []
        lines = query(
            self.conn,
            f"{_LINE_SELECT} WHERE pl.priceline_id IN ({_placeholders(priceline_ids)}) "
            "ORDER BY pl.priceline_id",
            priceline_ids,
        )
        for line in lines:
            surcharges = query(
                self.conn,
                "SELECT charge_code, amount_20st, amount_40st, amount_40hc "
                "FROM price_line_surcharges WHERE priceline_id = ? ORDER BY charge_code",
                [line["priceline_id"]],
            )
            line["surcharges"] = surcharges
            for size in SIZES:
                extra = (
                    0.0
                    if line["charges_included"]
                    else sum(s[f"amount_{size}"] for s in surcharges)
                )
                line[f"all_in_{size}"] = line[f"rate_{size}"] + extra
        return lines

    def _lines_on_route(self, route_id: int, commodity_id: int | None, status: str) -> list[Row]:
        sql = f"{_LINE_SELECT} WHERE pl.route_id = ? AND pl.status = ?"
        params: list[Any] = [route_id, status]
        if commodity_id is not None:
            sql += " AND pl.commodity_id = ?"
            params.append(commodity_id)
        rows = query(self.conn, sql + " ORDER BY pl.valid_from", params)
        return self._line_details([r["priceline_id"] for r in rows])

    @staticmethod
    def _overlaps(lines: list[Row], on_date: str) -> list[Row]:
        """ACTIVE lines for the same product (route, commodity, SOC/HAZ flags) valid on ``on_date``.

        A SOC or HAZ line next to a standard line on the same lane is a different product, not
        an overlap.
        """
        groups: dict[tuple[int, int, int, int], list[Row]] = {}
        for line in lines:
            if line["status"] == "ACTIVE" and line["valid_from"] <= on_date <= line["valid_to"]:
                key = (line["route_id"], line["commodity_id"], line["has_soc"], line["has_haz"])
                groups.setdefault(key, []).append(line)
        return [line for group in groups.values() if len(group) > 1 for line in group]

    def _inland(self, route: Row, on_date: str) -> dict[str, Any]:
        rates = query(
            self.conn,
            "SELECT inland_point, gateway_port, charge_code, amount_20, amount_40, valid_from, "
            "valid_to FROM inland_rates WHERE inland_point = ? AND gateway_port = ? "
            "AND valid_from <= ? AND valid_to >= ?",
            [route["pol_code"], route["via_port"], on_date, on_date],
        )
        catalog = query(
            self.conn,
            "SELECT charge_code, description, is_active FROM charges WHERE charge_type = 'INLAND'",
        )
        return {
            "inland_point": route["pol_code"],
            "inland_point_name": route["pol_name"],
            "gateway_port": route["via_port"],
            "pre_carriage_rate_found": any(r["charge_code"] == "INL00" for r in rates),
            "rates": rates,
            "inland_charge_catalog": catalog,
        }

    def run(self, ticket_id: int) -> DataEvidence:
        ticket = query_one(self.conn, "SELECT * FROM tickets WHERE ticket_id = ?", [ticket_id])
        if ticket is None:
            raise LookupError(f"ticket {ticket_id} not found")
        on_date = str(ticket["created_at"])[:10]
        refs = extract_references(str(ticket["description"]))

        bookings = []
        if refs["bookings"]:
            bookings = query(
                self.conn,
                "SELECT b.booking_ref, b.status, b.equipment_size, b.quantity, b.total_amount, "  # noqa: S608 - only ? placeholders are interpolated
                "b.booking_date, q.quote_ref FROM bookings b JOIN quotes q USING (quote_id) "
                f"WHERE b.booking_ref IN ({_placeholders(refs['bookings'])})",
                refs["bookings"],
            )
        quote_refs = sorted(set(refs["quotes"]) | {b["quote_ref"] for b in bookings})
        quotes = []
        if quote_refs:
            quotes = query(
                self.conn,
                "SELECT quote_ref, priceline_id, customer_name, equipment_size, quantity, "  # noqa: S608 - only ? placeholders are interpolated
                "quoted_rate, quote_date, valid_until, status, channel FROM quotes "
                f"WHERE quote_ref IN ({_placeholders(quote_refs)}) ORDER BY quote_date",
                quote_refs,
            )
            for q in quotes:
                q["expired_on_ticket_date"] = q["valid_until"] < on_date

        line_ids = {q["priceline_id"] for q in quotes}
        if ticket["priceline_id"]:
            line_ids.add(ticket["priceline_id"])
        for ref in refs["price_lines"]:
            code, _, n = ref.partition(" line ")
            row = query_one(
                self.conn,
                "SELECT pl.priceline_id FROM price_lines pl JOIN price_lists pls USING (pricelist_id) "
                "WHERE pls.pricelist_code = ? AND pl.line_ref = ?",
                [code, int(n)],
            )
            if row:
                line_ids.add(row["priceline_id"])
        price_lines = self._line_details(sorted(line_ids))

        # Same route (and commodity) context: what expired, what is current, what overlaps.
        contexts: set[tuple[int, int | None]] = {
            (ln["route_id"], ln["commodity_id"]) for ln in price_lines
        }
        if ticket["route_id"] and not any(r == ticket["route_id"] for r, _ in contexts):
            contexts.add((ticket["route_id"], None))
        expired: list[Row] = []
        current: list[Row] = []
        for route_id, commodity_id in sorted(contexts, key=str):
            expired += self._lines_on_route(route_id, commodity_id, "EXPIRED")
            current += self._lines_on_route(route_id, commodity_id, "ACTIVE")
        overlapping = self._overlaps(current, on_date)

        inland = None
        route_ids = {ln["route_id"] for ln in price_lines} | (
            {ticket["route_id"]} if ticket["route_id"] else set()
        )
        for route_id in sorted(route_ids):
            route = query_one(self.conn, "SELECT * FROM routes WHERE route_id = ?", [route_id])
            if route and route["is_inland"]:
                inland = self._inland(route, on_date)
                break

        similar = query(
            self.conn,
            "SELECT ticket_ref, status, created_at, substr(description, 1, 160) AS preview "
            "FROM tickets WHERE subcategory = ? AND ticket_id != ? AND status != 'RESOLVED' "
            "ORDER BY created_at",
            [ticket["subcategory"], ticket_id],
        )
        reporter = None
        if ticket["reported_by"]:
            reporter = query_one(
                self.conn,
                "SELECT full_name, role FROM users WHERE user_id = ?",
                [ticket["reported_by"]],
            )

        return DataEvidence(
            ticket={k: ticket[k] for k in ("ticket_ref", "category", "subcategory", "created_at")},
            references=refs,
            quotes=quotes,
            bookings=bookings,
            price_lines=price_lines,
            expired_lines=[ln for ln in expired if ln["priceline_id"] not in line_ids],
            current_lines=[ln for ln in current if ln["priceline_id"] not in line_ids],
            overlapping_lines=overlapping,
            inland=inland,
            similar_open_tickets=similar,
            reporter=reporter,
        )
