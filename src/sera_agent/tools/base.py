from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool


@dataclass(slots=True)
class ToolResult:
    success: bool
    output: str


class _RunAdapter(BaseTool):
    def __init__(self, tool: Any) -> None:
        super().__init__(name=tool.name, description=getattr(tool, "description", ""), func=tool.run)

    def invoke(self, arguments: dict[str, Any]) -> Any:
        result = self.func(arguments)
        if hasattr(result, "output"):
            return result.output
        return result


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: Any) -> None:
        if not isinstance(tool, BaseTool):
            tool = _RunAdapter(tool)
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def descriptions(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    def as_list(self) -> list[BaseTool]:
        return list(self._tools.values())
