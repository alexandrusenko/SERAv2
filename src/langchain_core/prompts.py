from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .runnables import RunnableSequence


@dataclass
class _Message:
    role: str
    content: str


class ChatPromptTemplate:
    def __init__(self, messages: list[tuple[str, str]]) -> None:
        self.messages = messages

    @classmethod
    def from_messages(cls, messages: list[tuple[str, str]]) -> "ChatPromptTemplate":
        return cls(messages)

    def __or__(self, other: Any) -> RunnableSequence:
        return RunnableSequence(self, other)

    def invoke(self, values: dict[str, Any]) -> list[_Message]:
        rendered: list[_Message] = []
        for role, template in self.messages:
            rendered.append(_Message(role=role, content=template.format(**values)))
        return rendered
