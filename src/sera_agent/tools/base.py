from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class ToolResult:
    success: bool
    output: str


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    args_schema: dict[str, Any]


class Tool(Protocol):
    name: str
    description: str

    def run(self, arguments: dict[str, object]) -> ToolResult:
        ...

    def schema(self) -> ToolSpec:
        ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]

    def descriptions(self) -> str:
        lines = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)

    def specs(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        for tool in self._tools.values():
            if hasattr(tool, "schema"):
                specs.append(tool.schema())
            else:
                specs.append(
                    ToolSpec(
                        name=tool.name,
                        description=tool.description,
                        args_schema={"type": "object", "properties": {}},
                    )
                )
        return specs

    def has(self, name: str) -> bool:
        return name in self._tools
