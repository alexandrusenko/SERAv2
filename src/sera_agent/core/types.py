from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class StepResult:
    step: str
    success: bool
    output: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(slots=True)
class MemoryItem:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
