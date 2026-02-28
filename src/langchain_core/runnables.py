from __future__ import annotations

from typing import Any, Callable


class RunnableSequence:
    def __init__(self, left: Any, right: Any) -> None:
        self.left = left
        self.right = right

    def __or__(self, other: Any) -> "RunnableSequence":
        return RunnableSequence(self, other)

    def invoke(self, value: Any) -> Any:
        intermediate = self.left.invoke(value)
        return self.right.invoke(intermediate)


class RunnableLambda:
    def __init__(self, func: Callable[[Any], Any]) -> None:
        self.func = func

    def __or__(self, other: Any) -> RunnableSequence:
        return RunnableSequence(self, other)

    def invoke(self, value: Any) -> Any:
        return self.func(value)
