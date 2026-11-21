"""SQLite access. Every query in the project goes through bound parameters."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hypercare.config import PACKAGE_DIR

SCHEMA_PATH = PACKAGE_DIR / "schema.sql"

Row = dict[str, Any]


def connect(path: Path | str) -> sqlite3.Connection:
    """Open a connection with dict-like rows and foreign keys enforced.

    ``check_same_thread=False`` lets the Streamlit app reuse one connection across reruns;
    writes are short and serialised by SQLite.
    """
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text())


def query(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> list[Row]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def query_one(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> Row | None:
    row = conn.execute(sql, params).fetchone()
    return dict(row) if row is not None else None


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def utcnow() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
