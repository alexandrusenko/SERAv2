from __future__ import annotations

from pathlib import Path

from sera_agent.config.models import AgentConfig, MemoryConfig, RuntimeConfig, SafetyConfig
from sera_agent.memory.store import MemoryStore
from sera_agent.orchestration.agent import SeraAgent
from sera_agent.tools.base import ToolRegistry, ToolResult


class DummyLLM:
    def complete_json(self, system: str, user: str, schema_hint: str, max_tokens: int = 1024) -> dict[str, object]:
        if '"steps"' in schema_hint:
            return {"steps": ["подготовить ответ"]}
        return {
            "thought": "noop",
            "tool_calls": [],
            "final_output": "Готово",
            "success": True,
        }


class DummyImprover:
    def attempt_create_missing_tool(self, missing_tool: str, task_context: str) -> None:
        return


def test_run_emits_realtime_events(tmp_path: Path) -> None:
    config = AgentConfig(
        runtime=RuntimeConfig(model_path=tmp_path / "model.gguf"),
        memory=MemoryConfig(db_path=tmp_path / "memory.sqlite3"),
        safety=SafetyConfig(working_dir=tmp_path),
    )
    agent = SeraAgent(
        config=config,
        llm=DummyLLM(),
        memory=MemoryStore(config.memory.db_path),
        tools=ToolRegistry(),
        improver=DummyImprover(),
    )

    events: list[dict[str, str]] = []
    final = agent.run("Сделай тест", event_handler=events.append)

    assert "Готово" in final
    kinds = {event["kind"] for event in events}
    assert "plan" in kinds
    assert "step" in kinds
    assert "final" in kinds


class ExplodingTool:
    name = "explode"
    description = "always fails"

    def run(self, arguments: dict[str, object]) -> ToolResult:
        raise RuntimeError("boom")


class ToolCallLLM:
    def complete_json(self, system: str, user: str, schema_hint: str, max_tokens: int = 1024) -> dict[str, object]:
        if '"steps"' in schema_hint:
            return {"steps": ["выполнить инструмент"]}
        return {
            "thought": "call exploding tool",
            "tool_calls": [{"name": "explode", "arguments": {}}],
            "final_output": "done",
            "success": True,
        }


def test_run_survives_tool_exception(tmp_path: Path) -> None:
    config = AgentConfig(
        runtime=RuntimeConfig(model_path=tmp_path / "model.gguf"),
        memory=MemoryConfig(db_path=tmp_path / "memory.sqlite3"),
        safety=SafetyConfig(working_dir=tmp_path),
    )
    tools = ToolRegistry()
    tools.register(ExplodingTool())
    agent = SeraAgent(
        config=config,
        llm=ToolCallLLM(),
        memory=MemoryStore(config.memory.db_path),
        tools=tools,
        improver=DummyImprover(),
    )

    final = agent.run("Сделай тест")

    assert "Tool error: boom" in final


class FailingLLM:
    def __init__(self) -> None:
        self.reflection_calls = 0

    def complete_json(self, system: str, user: str, schema_hint: str, max_tokens: int = 1024) -> dict[str, object]:
        if '"steps"' in schema_hint:
            return {"steps": ["первый шаг"]}
        if '"critique"' in schema_hint:
            self.reflection_calls += 1
            return {
                "critique": "Нужно добавить проверку после ошибки.",
                "next_steps": ["попробовать альтернативный подход"],
            }
        return {
            "thought": "failed",
            "tool_calls": [],
            "final_output": "ошибка",
            "success": False,
        }


def test_run_writes_scratchpad_and_reflects_on_failure(tmp_path: Path) -> None:
    llm = FailingLLM()
    config = AgentConfig(
        runtime=RuntimeConfig(model_path=tmp_path / "model.gguf"),
        memory=MemoryConfig(db_path=tmp_path / "memory.sqlite3"),
        safety=SafetyConfig(working_dir=tmp_path),
    )
    agent = SeraAgent(
        config=config,
        llm=llm,
        memory=MemoryStore(config.memory.db_path),
        tools=ToolRegistry(),
        improver=DummyImprover(),
    )

    events: list[dict[str, str]] = []
    agent.run("Сделай тест", event_handler=events.append)

    scratchpad = (tmp_path / "memory.md").read_text(encoding="utf-8")
    assert "## Dynamic Plan" in scratchpad
    assert "Reflection" in scratchpad
    assert llm.reflection_calls > 0
    assert any(event["kind"] == "reflection" for event in events)


class DirectAnswerLLM:
    def __init__(self) -> None:
        self.complete_calls = 0
        self.complete_json_calls = 0

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        self.complete_calls += 1
        return "Привет!"

    def complete_json(self, system: str, user: str, schema_hint: str, max_tokens: int = 1024) -> dict[str, object]:
        self.complete_json_calls += 1
        return {"steps": ["should not happen"]}


def test_simple_greeting_is_answered_without_plan(tmp_path: Path) -> None:
    llm = DirectAnswerLLM()
    config = AgentConfig(
        runtime=RuntimeConfig(model_path=tmp_path / "model.gguf"),
        memory=MemoryConfig(db_path=tmp_path / "memory.sqlite3"),
        safety=SafetyConfig(working_dir=tmp_path),
    )
    agent = SeraAgent(
        config=config,
        llm=llm,
        memory=MemoryStore(config.memory.db_path),
        tools=ToolRegistry(),
        improver=DummyImprover(),
    )

    events: list[dict[str, str]] = []
    final = agent.run("Привет", event_handler=events.append)

    assert final == "Привет!"
    assert llm.complete_calls == 1
    assert llm.complete_json_calls == 0
    assert not any(event["kind"] == "plan" for event in events)
