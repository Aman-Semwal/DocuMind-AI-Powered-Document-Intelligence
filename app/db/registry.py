"""SQLite registry (WAL mode): documents, query log, feedback, embedding cache."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from app.core.exceptions import DuplicateDocument, NotFound

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL UNIQUE,
    sha256 TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    pages INTEGER NOT NULL DEFAULT 0,
    pages_processed INTEGER NOT NULL DEFAULT 0,
    chunks INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    stage TEXT NOT NULL DEFAULT 'uploaded',
    error TEXT,
    resumable INTEGER NOT NULL DEFAULT 0,
    warnings TEXT NOT NULL DEFAULT '[]',
    summary TEXT,
    suggested_questions TEXT NOT NULL DEFAULT '[]',
    embed_model TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS query_log (
    question_id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    question TEXT NOT NULL,
    answered INTEGER NOT NULL,
    confidence TEXT NOT NULL,
    cached INTEGER NOT NULL DEFAULT 0,
    timings TEXT NOT NULL DEFAULT '{}',
    document_ids TEXT NOT NULL DEFAULT '[]',
    verification TEXT,
    citations TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS feedback (
    question_id TEXT PRIMARY KEY,
    rating TEXT NOT NULL,
    comment TEXT,
    ts REAL NOT NULL
);
"""

_JSON_FIELDS = ("warnings", "suggested_questions")
_UPDATABLE = frozenset(
    {
        "pages",
        "pages_processed",
        "chunks",
        "status",
        "stage",
        "error",
        "resumable",
        "warnings",
        "summary",
        "suggested_questions",
        "embed_model",
        "updated_at",
        "started_at",
    }
)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for f in _JSON_FIELDS:
        if f in d and isinstance(d[f], str):
            d[f] = json.loads(d[f])
    d["resumable"] = bool(d.get("resumable", 0))
    return d


class Registry:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)
            query_columns = {row["name"] for row in c.execute("PRAGMA table_info(query_log)")}
            if "document_ids" not in query_columns:
                c.execute("ALTER TABLE query_log ADD COLUMN document_ids TEXT NOT NULL DEFAULT '[]'")
            if "verification" not in query_columns:
                c.execute("ALTER TABLE query_log ADD COLUMN verification TEXT")
            if "citations" not in query_columns:
                c.execute("ALTER TABLE query_log ADD COLUMN citations TEXT NOT NULL DEFAULT '[]'")
            c.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('docset_version', '0')")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------ docs
    def create_document(
        self, *, sha256: str, filename: str, stored_path: str, size_bytes: int, pages: int
    ) -> dict[str, Any]:
        """Create a row and a sequential ID (DOC001...) inside one IMMEDIATE transaction."""
        now = time.time()
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                dup = c.execute("SELECT id FROM documents WHERE sha256 = ?", (sha256,)).fetchone()
                if dup:
                    raise DuplicateDocument(f"This file was already uploaded as {dup['id']}.", existing_id=dup["id"])
                seq = c.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM documents").fetchone()[0]
                doc_id = f"DOC{seq:03d}"
                c.execute(
                    "INSERT INTO documents (id, seq, sha256, filename, stored_path, size_bytes,"
                    " pages, status, stage, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', 'queued', ?, ?)",
                    (doc_id, seq, sha256, filename, stored_path, size_bytes, pages, now, now),
                )
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise
        return self.get(doc_id)

    def get(self, doc_id: str) -> dict[str, Any]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if row is None:
            raise NotFound(f"Document {doc_id} was not found.")
        return _row_to_dict(row)

    def exists(self, doc_id: str) -> bool:
        with self._conn() as c:
            return c.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone() is not None

    def list_documents(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM documents ORDER BY seq").fetchall()
        return [_row_to_dict(r) for r in rows]

    def completed_ids(self) -> list[str]:
        with self._conn() as c:
            rows = c.execute("SELECT id FROM documents WHERE status='completed' ORDER BY seq")
            return [r["id"] for r in rows.fetchall()]

    def count_active(self) -> int:
        with self._conn() as c:
            return int(
                c.execute("SELECT COUNT(*) FROM documents WHERE status IN ('pending','processing')").fetchone()[0]
            )

    def incomplete(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM documents WHERE status IN ('pending','processing') ORDER BY seq").fetchall()
        return [_row_to_dict(r) for r in rows]

    def update(self, doc_id: str, **fields: Any) -> None:
        if not fields:
            return
        for f in _JSON_FIELDS:
            if f in fields:
                fields[f] = json.dumps(fields[f])
        if "resumable" in fields:
            fields["resumable"] = int(bool(fields["resumable"]))
        fields["updated_at"] = time.time()
        unknown = set(fields) - _UPDATABLE
        if unknown:  # column names are interpolated below, so only known names are allowed
            raise ValueError(f"unknown document fields: {sorted(unknown)}")
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE documents SET {cols} WHERE id = ?", (*fields.values(), doc_id))  # noqa: S608  # nosec B608

    def add_warning(self, doc_id: str, message: str) -> None:
        doc = self.get(doc_id)
        warnings = [*doc["warnings"], message][-20:]
        self.update(doc_id, warnings=warnings)

    def delete(self, doc_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    def delete_document_insights(self, doc_id: str) -> int:
        """Permanently remove question and feedback records attributed to a document."""
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                question_ids = [
                    row["question_id"]
                    for row in c.execute("SELECT question_id, document_ids FROM query_log").fetchall()
                    if doc_id in json.loads(row["document_ids"])
                ]
                if question_ids:
                    placeholders = ",".join("?" for _ in question_ids)
                    c.execute(
                        f"DELETE FROM feedback WHERE question_id IN ({placeholders})",  # noqa: S608
                        question_ids,
                    )
                    c.execute(
                        f"DELETE FROM query_log WHERE question_id IN ({placeholders})",  # noqa: S608
                        question_ids,
                    )
                c.execute("COMMIT")
            except BaseException:
                c.execute("ROLLBACK")
                raise
        return len(question_ids)

    # ---------------------------------------------------------- doc-set version
    def docset_version(self) -> int:
        with self._conn() as c:
            row = c.execute("SELECT value FROM meta WHERE key='docset_version'").fetchone()
        return int(row["value"]) if row else 0

    def bump_version(self) -> int:
        with self._conn() as c:
            c.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key='docset_version'")
        return self.docset_version()

    # ----------------------------------------------------------- query log
    def new_question_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def log_query(
        self,
        *,
        question_id: str,
        question: str,
        answered: bool,
        confidence: str,
        cached: bool,
        timings: dict[str, float],
        document_ids: list[str] | None = None,
        verification: dict[str, Any] | None = None,
        citations: list[str] | None = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO query_log (question_id, ts, question, answered,"
                " confidence, cached, timings, document_ids, verification, citations)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    question_id,
                    time.time(),
                    question[:1000],
                    int(answered),
                    confidence,
                    int(cached),
                    json.dumps(timings),
                    json.dumps(document_ids or []),
                    json.dumps(verification) if verification is not None else None,
                    json.dumps(citations or []),
                ),
            )

    def recent_queries(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT q.*, f.rating AS rating FROM query_log q"
                " LEFT JOIN feedback f ON f.question_id = q.question_id"
                " ORDER BY q.ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["timings"] = json.loads(d["timings"])
            d["document_ids"] = json.loads(d["document_ids"])
            d["verification"] = json.loads(d["verification"]) if d["verification"] else None
            d["citations"] = json.loads(d["citations"])
            d["answered"] = bool(d["answered"])
            d["cached"] = bool(d["cached"])
            out.append(d)
        return out

    def query_exists(self, question_id: str) -> bool:
        with self._conn() as c:
            return c.execute("SELECT 1 FROM query_log WHERE question_id=?", (question_id,)).fetchone() is not None

    def set_feedback(self, question_id: str, rating: str, comment: str | None) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO feedback (question_id, rating, comment, ts) VALUES (?, ?, ?, ?)",
                (question_id, rating, comment, time.time()),
            )

    def feedback_counts(self, question_ids: list[str] | None = None) -> dict[str, int]:
        if question_ids is not None and not question_ids:
            return {"up": 0, "down": 0}
        with self._conn() as c:
            if question_ids is None:
                rows = c.execute("SELECT rating, COUNT(*) AS n FROM feedback GROUP BY rating")
            else:
                placeholders = ",".join("?" for _ in question_ids)
                query = (
                    "SELECT rating, COUNT(*) AS n FROM feedback "  # noqa: S608
                    f"WHERE question_id IN ({placeholders}) GROUP BY rating"
                )
                rows = c.execute(query, tuple(question_ids))
            counts = {r["rating"]: int(r["n"]) for r in rows.fetchall()}
        return {"up": counts.get("up", 0), "down": counts.get("down", 0)}


class EmbeddingCache:
    """Stores embeddings keyed by sha256(text + model) so identical chunks are never re-embedded."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
        finally:
            conn.close()

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        with self._conn() as c:
            for i in range(0, len(keys), 500):
                part = keys[i : i + 500]
                marks = ",".join("?" * len(part))
                for key, blob in c.execute(
                    f"SELECT key, vec FROM emb WHERE key IN ({marks})",  # noqa: S608  # nosec B608
                    part,
                ):
                    found[key] = np.frombuffer(blob, dtype=np.float32).tolist()
        return found

    def put_many(self, items: dict[str, list[float]]) -> None:
        if not items:
            return
        rows = [(k, np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()]
        with self._conn() as c:
            c.execute("BEGIN")
            c.executemany("INSERT OR REPLACE INTO emb (key, vec) VALUES (?, ?)", rows)
            c.execute("COMMIT")
