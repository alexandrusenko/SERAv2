from __future__ import annotations

from pathlib import Path

from sera_agent.core.types import MemoryItem
from sera_agent.memory.store import MemoryStore
from sera_agent.safety.policy import SafetyError, SafetyPolicy
from sera_agent.config.models import SafetyConfig


def test_memory_roundtrip(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "mem.sqlite3")
    store.add(MemoryItem(role="user", content="hello world"))
    rows = store.search("hello", limit=5)
    assert rows
    assert rows[0].content == "hello world"


def test_safe_path_blocks_traversal(tmp_path: Path) -> None:
    safety = SafetyPolicy(SafetyConfig(working_dir=tmp_path))
    try:
        safety.safe_path("../escape.txt")
    except SafetyError:
        return
    raise AssertionError("Path traversal was not blocked")
