from __future__ import annotations

import json
from typing import Any

from .runnables import RunnableSequence


class JsonOutputParser:
    def __or__(self, other: Any) -> RunnableSequence:
        return RunnableSequence(self, other)

    def invoke(self, value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value)
        return value


class StrOutputParser:
    def __or__(self, other: Any) -> RunnableSequence:
        return RunnableSequence(self, other)

    def invoke(self, value: Any) -> str:
        return str(value)
