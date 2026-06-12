"""SQLite storage for the human-eval webapp.

One row per (participant, problem) in ``responses``; participants identified by
email. WAL mode keeps concurrent readers/writers happy for a single-process
FastAPI server.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent / "eval.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS participants (
    email       TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    user_agent  TEXT
);

CREATE TABLE IF NOT EXISTS responses (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    email              TEXT NOT NULL,
    problem_id         TEXT NOT NULL,
    selected           TEXT NOT NULL,
    is_correct         INTEGER NOT NULL,
    client_elapsed_ms  INTEGER,
    hidden_ms          INTEGER DEFAULT 0,
    served_at          REAL,
    answered_at        REAL,
    position           INTEGER,
    flagged            INTEGER DEFAULT 0,
    revision           INTEGER DEFAULT 1,
    note               TEXT,
    created_at         REAL NOT NULL,
    UNIQUE (email, problem_id)
);

CREATE INDEX IF NOT EXISTS idx_responses_email ON responses (email);
CREATE INDEX IF NOT EXISTS idx_responses_problem ON responses (problem_id);
"""


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        # Migration: tag responses with the app revision so we can separate
        # cohorts (e.g. pre/post instructions+feedback). Existing rows predate
        # this column and default to revision 1.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(responses)")}
        if "revision" not in cols:
            conn.execute("ALTER TABLE responses ADD COLUMN revision INTEGER DEFAULT 1")
        if "note" not in cols:
            conn.execute("ALTER TABLE responses ADD COLUMN note TEXT")
        conn.commit()


def upsert_participant(conn: sqlite3.Connection, email: str, user_agent: str = "") -> None:
    conn.execute(
        "INSERT INTO participants (email, created_at, user_agent) VALUES (?, ?, ?) "
        "ON CONFLICT(email) DO NOTHING",
        (email, time.time(), user_agent),
    )
    conn.commit()


def answered_ids(conn: sqlite3.Connection, email: str) -> set[str]:
    rows = conn.execute(
        "SELECT problem_id FROM responses WHERE email = ?", (email,)
    ).fetchall()
    return {r["problem_id"] for r in rows}


def progress(conn: sqlite3.Connection, email: str) -> dict:
    row = conn.execute(
        "SELECT COUNT(*) AS done, "
        "COALESCE(SUM(client_elapsed_ms), 0) AS total_ms, "
        "COALESCE(SUM(is_correct), 0) AS correct "
        "FROM responses WHERE email = ?",
        (email,),
    ).fetchone()
    return {"done": row["done"], "total_ms": row["total_ms"], "correct": row["correct"]}


def record_answer(
    conn: sqlite3.Connection,
    *,
    email: str,
    problem_id: str,
    selected: str,
    is_correct: bool,
    client_elapsed_ms: Optional[int],
    hidden_ms: int,
    served_at: Optional[float],
    position: int,
    flagged: bool,
    revision: int,
    note: Optional[str] = None,
) -> bool:
    """Insert a response. Returns False if this (email, problem) already exists."""
    try:
        conn.execute(
            "INSERT INTO responses "
            "(email, problem_id, selected, is_correct, client_elapsed_ms, hidden_ms, "
            " served_at, answered_at, position, flagged, revision, note, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                email,
                problem_id,
                selected,
                int(is_correct),
                client_elapsed_ms,
                hidden_ms,
                served_at,
                time.time(),
                position,
                int(flagged),
                revision,
                note,
                time.time(),
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
