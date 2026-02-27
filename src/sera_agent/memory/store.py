from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from sera_agent.core.types import MemoryItem

LOGGER = logging.getLogger(__name__)


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def add(self, item: MemoryItem) -> None:
        LOGGER.debug("Memory add role=%s", item.role)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memories (role, content, timestamp) VALUES (?, ?, ?)",
                (item.role, item.content, item.timestamp),
            )
            conn.commit()

    def search(self, query: str, limit: int = 10) -> list[MemoryItem]:
        tokens = [t for t in query.strip().split() if t]
        if not tokens:
            return self.latest(limit)

        clauses = " OR ".join(["content LIKE ?" for _ in tokens])
        params = [f"%{token}%" for token in tokens]
        sql = (
            "SELECT role, content, timestamp FROM memories "
            f"WHERE {clauses} ORDER BY id DESC LIMIT ?"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, [*params, limit]).fetchall()
        return [MemoryItem(role=r[0], content=r[1], timestamp=r[2]) for r in rows]

    def latest(self, limit: int = 10) -> list[MemoryItem]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, timestamp FROM memories ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [MemoryItem(role=r[0], content=r[1], timestamp=r[2]) for r in rows]
