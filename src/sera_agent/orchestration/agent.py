from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.prebuilt import create_react_agent

from sera_agent.config.models import AgentConfig
from sera_agent.core.types import MemoryItem
from sera_agent.memory.store import MemoryStore
from sera_agent.runtime.llm_engine import LLMEngine
from sera_agent.self_improve.improver import SelfImprover
from sera_agent.tools.base import ToolRegistry

LOGGER = logging.getLogger(__name__)


class _LLMEngineChatModel(BaseChatModel):
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    @property
    def _llm_type(self) -> str:
        return "sera-llm-engine"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager
        system_parts: list[str] = []
        user_parts: list[str] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_parts.append(msg.content if isinstance(msg.content, str) else str(msg.content))
            elif isinstance(msg, HumanMessage):
                user_parts.append(msg.content if isinstance(msg.content, str) else str(msg.content))
            else:
                user_parts.append(f"[{msg.type}] {msg.content}")

        text = self.engine.complete(
            system="\n\n".join(system_parts),
            user="\n\n".join(user_parts),
            max_tokens=int(kwargs.get("max_tokens", 1024)),
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class SeraAgent:
    def _build_dialog_window(self) -> str:
        pairs = self.memory.last_qa_pairs(limit_pairs=5)
        if not pairs:
            return ""
        lines: list[str] = ["Контекст последних 5 диалогов (вопрос-ответ):"]
        for idx, (question, answer) in enumerate(pairs, start=1):
            lines.append(f"{idx}. Q: {question}")
            lines.append(f"   A: {answer}")
        return "\n".join(lines)

    def _build_semantic_memory_context(self, task: str) -> str:
        relevant = self.memory.search(task, limit=self.config.memory.max_results)
        long_term = [item for item in relevant if item.role == "long_term"]
        if not long_term:
            return ""
        lines: list[str] = ["Релевантные данные из долговременной семантической памяти:"]
        for idx, item in enumerate(long_term, start=1):
            lines.append(f"{idx}. {item.content}")
        return "\n".join(lines)

    def _is_simple_question(self, task: str) -> bool:
        return task.strip().lower() in {"привет", "hello", "hi"}

    def __init__(
        self,
        config: AgentConfig,
        llm: LLMEngine,
        memory: MemoryStore,
        tools: ToolRegistry,
        improver: SelfImprover,
    ) -> None:
        del improver
        self.config = config
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.model = _LLMEngineChatModel(engine=llm)
        self.graph = create_react_agent(
            model=self.model,
            tools=self.tools.as_list(),
            prompt=(
                "Ты SERA-агент. Работай строго в стиле ReAct через доступные tools. "
                "Если tools не нужны — дай прямой ответ."
            ),
        )

    def _emit(self, event_handler: Callable[[dict[str, str]], None] | None, kind: str, message: str) -> None:
        if event_handler:
            event_handler({"kind": kind, "message": message})

    def run(self, task: str, event_handler: Callable[[dict[str, str]], None] | None = None) -> str:
        self.memory.add(MemoryItem(role="user", content=task))
        self._emit(event_handler, "log", f"Старт задачи: {task}")

        if not task.strip():
            reflection = "Idle Reflection\nGoal: improve reasoning"
            self.memory.add(MemoryItem(role="assistant", content=reflection))
            self._emit(event_handler, "reflection", reflection)
            self._emit(event_handler, "final", reflection)
            return reflection

        if self._is_simple_question(task) and hasattr(self.llm, "complete"):
            final = self.llm.complete(system="Ты SERA", user=task, max_tokens=300)
            self.memory.add(MemoryItem(role="assistant", content=final))
            self._emit(event_handler, "final", final)
            return final

        self._emit(event_handler, "plan", "1. ReAct: analyze task and call tools if needed")
        dialog_window = self._build_dialog_window()
        semantic_context = self._build_semantic_memory_context(task)
        context_parts = [part for part in [dialog_window, semantic_context] if part]
        request_payload = task
        if context_parts:
            context_block = "\n\n".join(context_parts)
            request_payload = f"{context_block}\n\nТекущий запрос пользователя:\n{task}"
        state = self.graph.invoke({"messages": [("user", request_payload)]})
        messages = state.get("messages", []) if isinstance(state, dict) else []
        final = ""
        tool_outputs: list[str] = []
        for msg in messages:
            msg_type = getattr(msg, "type", "")
            content = getattr(msg, "content", "")
            if msg_type == "tool":
                text_out = str(content)
                tool_outputs.append(text_out)
                self._emit(event_handler, "step", text_out)
            if msg_type == "ai":
                final = str(content)

        if not tool_outputs:
            self._emit(event_handler, "step", "ReAct iteration completed")
        prefix = "\n".join(tool_outputs).strip()
        if prefix:
            final = f"{prefix}\n{final}".strip()
        final = final or "Задача завершена без текстового ответа."
        self.memory.add(MemoryItem(role="assistant", content=final))
        self._emit(event_handler, "final", final)
        return final
