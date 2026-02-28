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


class IdleLLM:
    def complete_json(self, system: str, user: str, schema_hint: str, max_tokens: int = 1024) -> dict[str, object]:
        return {
            "learning_goal": "Изучить устойчивые паттерны отладки",
            "experiment_plan": ["Смоделировать ошибку импорта", "Проверить автокоррекцию"],
            "self_test": "assert 1 + 1 == 2",
        }


def test_empty_task_triggers_idle_metacognition(tmp_path: Path) -> None:
    config = AgentConfig(
        runtime=RuntimeConfig(model_path=tmp_path / "model.gguf"),
        memory=MemoryConfig(db_path=tmp_path / "memory.sqlite3"),
        safety=SafetyConfig(working_dir=tmp_path),
    )
    agent = SeraAgent(
        config=config,
        llm=IdleLLM(),
        memory=MemoryStore(config.memory.db_path),
        tools=ToolRegistry(),
        improver=DummyImprover(),
    )

    events: list[dict[str, str]] = []
    final = agent.run("", event_handler=events.append)

    assert "Idle Reflection" in final
    assert any(event["kind"] == "reflection" for event in events)


class ShellRecorderTool:
    name = "shell"
    description = "records install command"

    def __init__(self) -> None:
        self.commands: list[str] = []

    def run(self, arguments: dict[str, object]) -> ToolResult:
        command = str(arguments.get("command", ""))
        self.commands.append(command)
        return ToolResult(True, "installed")


class AutoFixLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, system: str, user: str, schema_hint: str, max_tokens: int = 1024) -> dict[str, object]:
        if '"steps"' in schema_hint:
            return {"steps": ["запустить python-скрипт"]}
        if '"critique"' in schema_hint:
            return {"critique": "Не хватает пакета", "next_steps": []}
        self.calls += 1
        if self.calls == 1:
            return {
                "thought": "attempt 1",
                "tool_calls": [],
                "final_output": "ModuleNotFoundError: No module named 'some_lib'",
                "success": False,
            }
        return {
            "thought": "attempt 2",
            "tool_calls": [],
            "final_output": "Готово",
            "success": True,
        }


class SummarizingLLM:
    def complete_json(self, system: str, user: str, schema_hint: str, max_tokens: int = 1024) -> dict[str, object]:
        if '"steps"' in schema_hint:
            return {"steps": ["сделать поиск"]}
        return {
            "thought": "done",
            "tool_calls": [],
            "final_output": "STEP 1 internal",
            "success": True,
        }

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        assert "Agent execution transcript" in user
        return "Итоговый ответ без служебных шагов"


def test_run_auto_installs_missing_dependency_via_shell(tmp_path: Path) -> None:
    config = AgentConfig(
        runtime=RuntimeConfig(model_path=tmp_path / "model.gguf"),
        memory=MemoryConfig(db_path=tmp_path / "memory.sqlite3"),
        safety=SafetyConfig(working_dir=tmp_path),
    )
    shell = ShellRecorderTool()
    tools = ToolRegistry()
    tools.register(shell)
    agent = SeraAgent(
        config=config,
        llm=AutoFixLLM(),
        memory=MemoryStore(config.memory.db_path),
        tools=tools,
        improver=DummyImprover(),
    )

    final = agent.run("Поставь библиотеку и выполни код")

    assert "Готово" in final
    assert shell.commands
    assert shell.commands[0] == "python -m pip install some_lib"


def test_run_returns_single_summarized_final_message(tmp_path: Path) -> None:
    config = AgentConfig(
        runtime=RuntimeConfig(model_path=tmp_path / "model.gguf"),
        memory=MemoryConfig(db_path=tmp_path / "memory.sqlite3"),
        safety=SafetyConfig(working_dir=tmp_path),
    )
    agent = SeraAgent(
        config=config,
        llm=SummarizingLLM(),
        memory=MemoryStore(config.memory.db_path),
        tools=ToolRegistry(),
        improver=DummyImprover(),
    )

    events: list[dict[str, str]] = []
    final = agent.run("Найди новости", event_handler=events.append)

    assert final == "Итоговый ответ без служебных шагов"
    assert any(event["kind"] == "final" and event["message"] == final for event in events)


class ContextCaptureLLM:
    def __init__(self) -> None:
        self.users: list[str] = []

    def complete_json(self, system: str, user: str, schema_hint: str, max_tokens: int = 1024) -> dict[str, object]:
        del system, schema_hint, max_tokens
        self.users.append(user)
        return {
            "thought": "done",
            "tool_calls": [],
            "final_output": "ok",
            "success": True,
        }


def test_agent_passes_sliding_window_of_last_five_qa_pairs(tmp_path: Path) -> None:
    llm = ContextCaptureLLM()
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

    for i in range(6):
        agent.run(f"Вопрос {i}")

    payload = llm.users[-1]
    assert "Контекст последних 5 диалогов" in payload
    assert "Q: Вопрос 0" in payload
    assert "Q: Вопрос 4" in payload
    assert "Q: Вопрос 5" not in payload
    assert "Текущий запрос пользователя:\nВопрос 5" in payload


def test_agent_passes_relevant_long_term_semantic_memory_in_context(tmp_path: Path) -> None:
    llm = ContextCaptureLLM()
    config = AgentConfig(
        runtime=RuntimeConfig(model_path=tmp_path / "model.gguf"),
        memory=MemoryConfig(
            db_path=tmp_path / "memory.sqlite3",
            long_term_chunk_size=2,
            short_term_search_limit=50,
            long_term_search_limit=50,
        ),
        safety=SafetyConfig(working_dir=tmp_path),
    )
    agent = SeraAgent(
        config=config,
        llm=llm,
        memory=MemoryStore(
            config.memory.db_path,
            long_term_chunk_size=config.memory.long_term_chunk_size,
            short_term_search_limit=config.memory.short_term_search_limit,
            long_term_search_limit=config.memory.long_term_search_limit,
        ),
        tools=ToolRegistry(),
        improver=DummyImprover(),
    )

    agent.run("Сохрани заметку: python asyncio event loop")
    agent.run("Сохрани заметку: kafka stream consumer lag")
    agent.run("Что важно про asyncio loop?")

    payload = llm.users[-1]
    assert "Релевантные данные из долговременной семантической памяти" in payload
    assert "semantic_keys" in payload
    assert "Текущий запрос пользователя:\nЧто важно про asyncio loop?" in payload
