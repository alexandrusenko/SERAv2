from dataclasses import dataclass
from langchain_core.messages import AIMessage


@dataclass
class ChatGeneration:
    message: AIMessage


@dataclass
class ChatResult:
    generations: list[ChatGeneration]
