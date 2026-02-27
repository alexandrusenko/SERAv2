from __future__ import annotations

import sqlite3
from pathlib import Path

from sera_agent.config.models import SafetyConfig
from sera_agent.core.types import MemoryItem
from sera_agent.memory.store import MemoryStore
from sera_agent.safety.policy import SafetyError, SafetyPolicy


def test_memory_roundtrip(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "mem.sqlite3")
    store.add(MemoryItem(role="user", content="hello world"))
    rows = store.search("hello", limit=5)
    assert rows
    assert rows[0].content == "hello world"


def test_long_term_memory_compaction_and_search(tmp_path: Path) -> None:
    db_path = tmp_path / "mem.sqlite3"
    store = MemoryStore(db_path, long_term_chunk_size=2)
    store.add(MemoryItem(role="user", content="deploy payment service to staging"))
    store.add(MemoryItem(role="assistant", content="payment staging deployment prepared"))

    with sqlite3.connect(db_path) as conn:
        compacted = conn.execute("SELECT COUNT(*) FROM long_term_memories").fetchone()[0]
    assert compacted == 1

    rows = store.search("payment staging", limit=5)
    assert rows
    assert any("payment" in row.content.lower() for row in rows)
    assert any(row.role == "long_term" for row in rows)


def test_safe_path_blocks_traversal(tmp_path: Path) -> None:
    safety = SafetyPolicy(SafetyConfig(working_dir=tmp_path))
    try:
        safety.safe_path("../escape.txt")
    except SafetyError:
        return
    raise AssertionError("Path traversal was not blocked")
