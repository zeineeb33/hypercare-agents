"""Build the SQLite database from the synthetic seed data and the knowledge base."""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from hypercare.db import connect, create_schema, transaction

# Insertion order respects foreign keys.
CSV_TABLES = [
    "users",
    "pricing_groups",
    "routes",
    "commodities",
    "charges",
    "inland_rates",
    "price_lists",
    "price_lines",
    "price_line_surcharges",
    "voyages",
    "quotes",
    "bookings",
]

TICKET_COLUMNS = [
    "ticket_id",
    "ticket_ref",
    "category",
    "subcategory",
    "priority",
    "status",
    "description",
    "priceline_id",
    "route_id",
    "reported_by",
    "created_at",
    "related_doc",
    "scenario_id",
]


@dataclass(frozen=True)
class KbDocument:
    file_name: str
    title: str
    category: str
    version: str
    effective_date: str
    source_type: str
    content: str


def ticket_ref(ticket_id: int) -> str:
    return f"HC-2026-{ticket_id:04d}"


def parse_kb_document(path: Path) -> KbDocument:
    """Parse a Markdown file with a ``---`` front matter of ``key: value`` lines."""
    text = path.read_text()
    if not text.startswith("---\n"):
        raise ValueError(f"{path.name}: missing front matter")
    header, body = text[4:].split("\n---\n", 1)
    meta: dict[str, str] = {}
    for line in header.splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip().strip('"')
    missing = {"title", "category", "version", "effective_date", "source_type"} - meta.keys()
    if missing:
        raise ValueError(f"{path.name}: missing front matter keys {sorted(missing)}")
    return KbDocument(
        file_name=path.name,
        title=meta["title"],
        category=meta["category"],
        version=meta["version"],
        effective_date=meta["effective_date"],
        source_type=meta["source_type"],
        content=body.strip(),
    )


def load_kb_documents(kb_dir: Path) -> list[KbDocument]:
    return [parse_kb_document(p) for p in sorted(kb_dir.glob("*.md"))]


def _insert_csv(conn: sqlite3.Connection, table: str, path: Path) -> int:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0
    columns = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",  # noqa: S608 - fixed table list
        [[(value if value != "" else None) for value in row.values()] for row in rows],
    )
    return len(rows)


def seed_database(conn: sqlite3.Connection, seed_dir: Path, kb_dir: Path) -> dict[str, int]:
    """Create the schema and load everything. Returns the row count per table."""
    counts: dict[str, int] = {}
    with transaction(conn):
        create_schema(conn)
        for table in CSV_TABLES:
            counts[table] = _insert_csv(conn, table, seed_dir / f"{table}.csv")

        tickets = json.loads((seed_dir / "tickets.json").read_text())
        for t in tickets:
            t["ticket_ref"] = ticket_ref(t["ticket_id"])
        conn.executemany(
            f"INSERT INTO tickets ({', '.join(TICKET_COLUMNS)}) "  # noqa: S608 - fixed column list
            f"VALUES ({', '.join('?' for _ in TICKET_COLUMNS)})",
            [[t.get(c) for c in TICKET_COLUMNS] for t in tickets],
        )
        counts["tickets"] = len(tickets)

        docs = load_kb_documents(kb_dir)
        conn.executemany(
            "INSERT INTO kb_documents (file_name, title, category, version, effective_date, "
            "source_type, content) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    d.file_name,
                    d.title,
                    d.category,
                    d.version,
                    d.effective_date,
                    d.source_type,
                    d.content,
                )
                for d in docs
            ],
        )
        counts["kb_documents"] = len(docs)
    return counts


def build_database(db_path: Path, seed_dir: Path, kb_dir: Path) -> dict[str, int]:
    """Recreate the database file from scratch."""
    if db_path.exists():
        db_path.unlink()
    conn = connect(db_path)
    try:
        return seed_database(conn, seed_dir, kb_dir)
    finally:
        conn.close()
