from __future__ import annotations

import asyncio
from pathlib import Path

from sera_agent.ui import server


class _DummyLLM:
    def __init__(self) -> None:
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1


class _DummyAgent:
    def __init__(self) -> None:
        self.llm = _DummyLLM()
        self.run_calls: list[str] = []

    def run(self, task: str, event_handler=None) -> str:  # noqa: ANN001
        self.run_calls.append(task)
        if event_handler is not None:
            event_handler({"kind": "final", "message": f"done:{task}"})
        return f"done:{task}"


async def _collect_streaming_body(response) -> str:  # noqa: ANN001
    chunks: list[str] = []
    async for item in response.body_iterator:
        chunks.append(item)
    return "".join(chunks)


def test_ui_reuses_initialized_agent_for_multiple_requests(monkeypatch) -> None:  # noqa: ANN001
    built_agents: list[_DummyAgent] = []

    def fake_build_agent(_: Path) -> _DummyAgent:
        agent = _DummyAgent()
        built_agents.append(agent)
        return agent

    monkeypatch.setattr(server, "build_agent", fake_build_agent)

    server.initialize_agent()
    first = asyncio.run(server.run(server.RunRequest(task="first")))
    second = asyncio.run(server.run(server.RunRequest(task="second")))
    first_payload = asyncio.run(_collect_streaming_body(first))
    second_payload = asyncio.run(_collect_streaming_body(second))

    assert len(built_agents) == 1
    assert built_agents[0].llm.load_calls == 1
    assert built_agents[0].run_calls == ["first", "second"]
    assert "done:first" in first_payload
    assert "done:second" in second_payload
