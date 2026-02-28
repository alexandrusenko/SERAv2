from dataclasses import dataclass


@dataclass
class BaseMessage:
    content: str
    type: str


@dataclass
class HumanMessage(BaseMessage):
    type: str = "human"


@dataclass
class SystemMessage(BaseMessage):
    type: str = "system"


@dataclass
class AIMessage(BaseMessage):
    type: str = "ai"


@dataclass
class ToolMessage(BaseMessage):
    type: str = "tool"
