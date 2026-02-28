from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from sera_agent.config.models import AgentConfig
from sera_agent.core.logging import configure_logging
from sera_agent.memory.store import MemoryStore
from sera_agent.orchestration.agent import SeraAgent
from sera_agent.runtime.llm_engine import LLMEngine
from sera_agent.safety.policy import SafetyPolicy
from sera_agent.self_improve.improver import SelfImprover
from sera_agent.tools.base import ToolRegistry
from sera_agent.tools.builtin import FetchUrlTool, ReadFileTool, ShellTool, WebSearchTool, WriteFileTool
from sera_agent.tools.dynamic_loader import DynamicToolLoader

LOGGER = logging.getLogger(__name__)


def build_agent(config_path: Path) -> SeraAgent:
    config = AgentConfig.from_yaml(config_path)
    configure_logging(level=config.log_level, log_file=Path("logs/sera-agent.log"))

    safety = SafetyPolicy(config.safety)
    llm = LLMEngine(config.runtime)
    memory = MemoryStore(
        config.memory.db_path,
        long_term_chunk_size=config.memory.long_term_chunk_size,
        short_term_search_limit=config.memory.short_term_search_limit,
        long_term_search_limit=config.memory.long_term_search_limit,
    )
    registry = ToolRegistry()
    registry.register(ReadFileTool(safety=safety))
    registry.register(WriteFileTool(safety=safety))
    if config.safety.allow_shell:
        registry.register(ShellTool(safety=safety))
    registry.register(WebSearchTool(safety=safety))
    registry.register(FetchUrlTool(safety=safety))

    loader = DynamicToolLoader(tools_dir=Path("dynamic_tools"), safety=safety, registry=registry)
    improver = SelfImprover(llm=llm, loader=loader)
    return SeraAgent(config=config, llm=llm, memory=memory, tools=registry, improver=improver)


def main() -> None:
    parser = argparse.ArgumentParser(description="SERA local self-improving agent")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--serve", action="store_true", help="Run HTTP UI service")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("task", nargs="?", type=str)
    args = parser.parse_args()

    if args.serve:
        uvicorn.run("sera_agent.ui.server:app", host=args.host, port=args.port, reload=False)
        return

    agent = build_agent(args.config)
    if args.task:
        try:
            result = agent.run(args.task)
            print(result)
        except RuntimeError as exc:
            LOGGER.error("Agent failed to start or run: %s", exc)
            print(f"ERROR: {exc}")
            raise SystemExit(2) from exc
        return

    print("SERA interactive mode. Enter task (empty line or 'exit' to quit).")
    while True:
        try:
            task = input("sera> ").strip()
        except EOFError:
            print()
            break
        if not task or task.lower() in {"exit", "quit"}:
            break
        try:
            print(agent.run(task))
        except RuntimeError as exc:
            LOGGER.error("Agent failed to start or run: %s", exc)
            print(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
