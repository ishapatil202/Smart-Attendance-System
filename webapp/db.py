"""SQLite storage: students, face embeddings, attendance records.

Photos are never written anywhere. Only 128-number embedding vectors are kept,
and visitor enrollments are removed automatically after a retention window.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    is_sample   INTEGER NOT NULL DEFAULT 0,   -- bundled demo person, never auto-cleared
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    vector      BLOB NOT NULL,                -- 128 x float32, L2-normalised
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attendance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    date        TEXT NOT NULL,
    time        TEXT NOT NULL,
    confidence  REAL NOT NULL,
    UNIQUE (student_id, date)                 -- one check-in per student per day
);
CREATE INDEX IF NOT EXISTS idx_embeddings_student ON embeddings(student_id);
CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
"""


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self):
        self._conn.close()

    def _run(self, sql, params=()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _all(self, sql, params=()):
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    # ----------------------------------------------------------- enrollment
    def add_student(self, name: str, vectors: list[np.ndarray], is_sample: bool = False, created_at: str | None = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO students (name, is_sample, created_at) VALUES (?,?,?)",
                (name, int(is_sample), created_at or now_iso()),
            )
            sid = cur.lastrowid
            self._conn.executemany(
                "INSERT INTO embeddings (student_id, vector, created_at) VALUES (?,?,?)",
                [(sid, np.asarray(v, dtype="float32").tobytes(), now_iso()) for v in vectors],
            )
            self._conn.commit()
        return sid

    def sample_id(self, name: str) -> int | None:
        rows = self._all("SELECT id FROM students WHERE is_sample = 1 AND name = ?", (name,))
        return rows[0]["id"] if rows else None

    def gallery(self):
        """All stored embeddings as (student_ids[N], names[N], matrix[N,128])."""
        rows = self._all(
            "SELECT e.student_id, s.name, e.vector FROM embeddings e JOIN students s ON s.id = e.student_id"
        )
        if not rows:
            return [], [], np.zeros((0, 128), dtype="float32")
        matrix = np.stack([np.frombuffer(r["vector"], dtype="float32") for r in rows])
        return [r["student_id"] for r in rows], [r["name"] for r in rows], matrix

    def counts(self) -> dict:
        return self._all(
            "SELECT (SELECT COUNT(*) FROM students) AS students,"
            " (SELECT COUNT(*) FROM embeddings) AS embeddings,"
            " (SELECT COUNT(*) FROM attendance) AS records"
        )[0]

    def purge_visitors(self, older_than_hours: int) -> int:
        cutoff = (datetime.now() - timedelta(hours=older_than_hours)).isoformat(timespec="seconds")
        return self._run("DELETE FROM students WHERE is_sample = 0 AND created_at < ?", (cutoff,)).rowcount

    # ----------------------------------------------------------- attendance
    def mark_present(self, student_id: int, confidence: float, when: datetime | None = None) -> bool:
        """Record a check-in. Returns False if the student was already marked that day."""
        when = when or datetime.now()
        cur = self._run(
            "INSERT OR IGNORE INTO attendance (student_id, date, time, confidence) VALUES (?,?,?,?)",
            (student_id, when.date().isoformat(), when.strftime("%H:%M:%S"), round(float(confidence), 4)),
        )
        return cur.rowcount == 1

    def attendance(self, day: str | None = None, limit: int = 500) -> list[dict]:
        where, params = ("WHERE a.date = ?", (day,)) if day else ("", ())
        return self._all(
            "SELECT a.id, s.name, s.is_sample, a.date, a.time, a.confidence"
            f" FROM attendance a JOIN students s ON s.id = a.student_id {where}"
            " ORDER BY a.date DESC, a.time DESC LIMIT ?",
            (*params, limit),
        )
