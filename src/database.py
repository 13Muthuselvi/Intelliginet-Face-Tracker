"""SQLite persistence for registered faces, entry/exit events, and unique counts.

WAL mode keeps writes resilient if the process is interrupted mid-stream.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


class FaceDatabase:
    """Thread-safe SQLite wrapper used by the pipeline and the dashboard."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS faces (
                    face_id TEXT PRIMARY KEY,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    thumbnail_path TEXT
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id TEXT NOT NULL,
                    event_type TEXT NOT NULL CHECK (event_type IN ('entry', 'exit')),
                    timestamp TEXT NOT NULL,
                    image_path TEXT,
                    track_id INTEGER,
                    frame_index INTEGER,
                    FOREIGN KEY (face_id) REFERENCES faces(face_id)
                );

                CREATE INDEX IF NOT EXISTS idx_events_face ON events(face_id);
                CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
                """
            )
            self._conn.commit()

    def register_face(
        self,
        face_id: str,
        embedding: bytes,
        timestamp: datetime,
        thumbnail_path: str | None,
    ) -> None:
        ts = timestamp.isoformat(timespec="seconds")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO faces (face_id, first_seen, last_seen, embedding, thumbnail_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (face_id, ts, ts, embedding, thumbnail_path),
            )
            self._conn.commit()

    def touch_face(self, face_id: str, timestamp: datetime) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE faces SET last_seen = ? WHERE face_id = ?",
                (timestamp.isoformat(timespec="seconds"), face_id),
            )
            self._conn.commit()

    def insert_event(
        self,
        face_id: str,
        event_type: str,
        timestamp: datetime,
        image_path: str | None,
        track_id: int | None,
        frame_index: int | None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO events
                    (face_id, event_type, timestamp, image_path, track_id, frame_index)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    face_id,
                    event_type,
                    timestamp.isoformat(timespec="seconds"),
                    image_path,
                    track_id,
                    frame_index,
                ),
            )
            self._conn.commit()

    def load_gallery(self) -> list[tuple[str, bytes]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT face_id, embedding FROM faces"
            ).fetchall()
        return [(row["face_id"], row["embedding"]) for row in rows]

    def unique_visitor_count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM faces").fetchone()
        return int(row["n"]) if row else 0

    def clear_events(self) -> None:
        """Remove ENTRY/EXIT rows but keep registered faces for re-identification."""
        with self._lock:
            self._conn.execute("DELETE FROM events")
            self._conn.commit()

    def event_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT event_type, COUNT(*) AS n FROM events GROUP BY event_type"
            ).fetchall()
        counts = {"entry": 0, "exit": 0}
        for row in rows:
            counts[row["event_type"]] = int(row["n"])
        return counts

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, face_id, event_type, timestamp, image_path, track_id, frame_index
                FROM events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_faces(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT face_id, first_seen, last_seen, thumbnail_path
                FROM faces
                ORDER BY first_seen
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
