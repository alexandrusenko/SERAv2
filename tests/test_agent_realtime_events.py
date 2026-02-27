from __future__ import annotations

from pathlib import Path

from sera_agent.config.models import AgentConfig, MemoryConfig, RuntimeConfig, SafetyConfig
from sera_agent.memory.store import MemoryStore
from sera_agent.orchestration.agent import SeraAgent
from sera_agent.tools.base import ToolRegistry


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
