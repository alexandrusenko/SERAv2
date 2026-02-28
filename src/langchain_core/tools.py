from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class BaseTool:
    name: str
    description: str
    func: Callable[..., Any]

    def invoke(self, arguments: dict[str, Any]) -> Any:
        return self.func(**arguments)


def tool(name: str) -> Callable[[Callable[..., Any]], BaseTool]:
    def decorator(func: Callable[..., Any]) -> BaseTool:
        return BaseTool(name=name, description=(func.__doc__ or "").strip(), func=func)

    return decorator
