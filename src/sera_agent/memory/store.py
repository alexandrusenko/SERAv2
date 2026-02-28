from __future__ import annotations

import logging
import re
import sqlite3
from collections import Counter
from pathlib import Path

from sera_agent.core.types import MemoryItem

LOGGER = logging.getLogger(__name__)
_WORD_RE = re.compile(r"[\w-]+", flags=re.UNICODE)
_STOPWORDS = {
    "и",
    "в",
    "на",
    "с",
    "по",
    "для",
    "что",
    "это",
    "как",
    "или",
    "а",
    "но",
    "к",
    "из",
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
}


class MemoryStore:
    def __init__(
        self,
        db_path: Path,
        *,
        long_term_chunk_size: int = 8,
        short_term_search_limit: int = 40,
        long_term_search_limit: int = 40,
    ) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.long_term_chunk_size = max(1, long_term_chunk_size)
        self.short_term_search_limit = max(5, short_term_search_limit)
        self.long_term_search_limit = max(5, long_term_search_limit)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS short_term_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary TEXT NOT NULL,
                    semantic_keys TEXT NOT NULL,
                    source_from_id INTEGER NOT NULL,
                    source_to_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO memory_meta (key, value) VALUES ('last_compacted_id', '0')"
            )
            conn.commit()

    def _tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in _WORD_RE.findall(text) if len(t) > 1]

    def _score_text(self, text: str, tokens: list[str]) -> int:
        if not tokens:
            return 1
        haystack = text.lower()
        return sum(1 for token in tokens if token in haystack)

    def _get_last_compacted_id(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT value FROM memory_meta WHERE key = 'last_compacted_id'").fetchone()
        return int(row[0]) if row else 0

    def _set_last_compacted_id(self, conn: sqlite3.Connection, value: int) -> None:
        conn.execute(
            "UPDATE memory_meta SET value = ? WHERE key = 'last_compacted_id'",
            (str(value),),
        )

    def _summarize_batch(self, rows: list[tuple[int, str, str, str]]) -> tuple[str, str]:
        lines = [f"[{role}] {content.strip()}" for _, role, content, _ in rows]
        summary = " | ".join(lines[:6])
        if len(lines) > 6:
            summary += " | ..."

        token_counter: Counter[str] = Counter()
        for _, _, content, _ in rows:
            token_counter.update(t for t in self._tokenize(content) if t not in _STOPWORDS)
        semantic_keys = ", ".join(token for token, _ in token_counter.most_common(10))
        return summary, semantic_keys

    def _compact_if_needed(self, conn: sqlite3.Connection) -> None:
        last_compacted_id = self._get_last_compacted_id(conn)
        pending = conn.execute(
            "SELECT COUNT(*) FROM short_term_memories WHERE id > ?",
            (last_compacted_id,),
        ).fetchone()[0]
        if pending < self.long_term_chunk_size:
            return

        rows = conn.execute(
            """
            SELECT id, role, content, timestamp
            FROM short_term_memories
            WHERE id > ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (last_compacted_id, self.long_term_chunk_size),
        ).fetchall()
        if not rows:
            return

        summary, semantic_keys = self._summarize_batch(rows)
        from_id, to_id = rows[0][0], rows[-1][0]
        created_at = rows[-1][3]
        conn.execute(
            """
            INSERT INTO long_term_memories (
                summary, semantic_keys, source_from_id, source_to_id, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (summary, semantic_keys, from_id, to_id, created_at),
        )
        self._set_last_compacted_id(conn, to_id)
        LOGGER.debug("Compacted short-term memory batch %s..%s", from_id, to_id)

    def add(self, item: MemoryItem) -> None:
        LOGGER.debug("Memory add role=%s", item.role)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO short_term_memories (role, content, timestamp) VALUES (?, ?, ?)",
                (item.role, item.content, item.timestamp),
            )
            self._compact_if_needed(conn)
            conn.commit()

    def compact(self) -> None:
        with self._connect() as conn:
            self._compact_if_needed(conn)
            conn.commit()

    def search(self, query: str, limit: int = 10) -> list[MemoryItem]:
        tokens = [t for t in self._tokenize(query) if t]
        with self._connect() as conn:
            short_rows = conn.execute(
                """
                SELECT role, content, timestamp
                FROM short_term_memories
                ORDER BY id DESC
                LIMIT ?
                """,
                (self.short_term_search_limit,),
            ).fetchall()
            long_rows = conn.execute(
                """
                SELECT summary, semantic_keys, created_at
                FROM long_term_memories
                ORDER BY id DESC
                LIMIT ?
                """,
                (self.long_term_search_limit,),
            ).fetchall()

        scored: list[tuple[int, MemoryItem]] = []
        for role, content, ts in short_rows:
            score = self._score_text(content, tokens)
            if score > 0:
                scored.append((score + 2, MemoryItem(role=role, content=content, timestamp=ts)))

        for summary, semantic_keys, created_at in long_rows:
            score = self._score_text(summary, tokens) + self._score_text(semantic_keys, tokens)
            if score > 0:
                content = f"{summary}\n[semantic_keys] {semantic_keys}".strip()
                scored.append((score, MemoryItem(role="long_term", content=content, timestamp=created_at)))

        scored.sort(key=lambda pair: (pair[0], pair[1].timestamp), reverse=True)
        if not scored:
            return self.latest(limit)
        return [item for _, item in scored[:limit]]


    def last_qa_pairs(self, limit_pairs: int = 5) -> list[tuple[str, str]]:
        """Return the last N (user, assistant) dialogue pairs in chronological order."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM short_term_memories
                ORDER BY id ASC
                """
            ).fetchall()

        pairs: list[tuple[str, str]] = []
        pending_user: str | None = None
        for role, content in rows:
            if role == "user":
                pending_user = str(content)
                continue
            if role == "assistant" and pending_user is not None:
                pairs.append((pending_user, str(content)))
                pending_user = None

        if limit_pairs <= 0:
            return []
        return pairs[-limit_pairs:]

    def latest(self, limit: int = 10) -> list[MemoryItem]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, timestamp
                FROM short_term_memories
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [MemoryItem(role=r[0], content=r[1], timestamp=r[2]) for r in rows]
