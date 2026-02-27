from __future__ import annotations

import logging
from textwrap import dedent

from sera_agent.runtime.llm_engine import LLMEngine
from sera_agent.tools.dynamic_loader import DynamicToolLoader

LOGGER = logging.getLogger(__name__)


class SelfImprover:
    def __init__(self, llm: LLMEngine, loader: DynamicToolLoader) -> None:
        self.llm = llm
        self.loader = loader

    def attempt_create_missing_tool(self, missing_tool: str, task_context: str) -> str:
        LOGGER.warning("Attempting self-improvement for missing tool: %s", missing_tool)
        system = dedent(
            """
            You are generating a secure python tool plugin.
            Output only python code.
            Requirements:
            - define class implementing .name, .description, .run(arguments)
            - define function build_tool() returning tool instance
            - no dangerous imports (os, subprocess, ctypes, socket, shutil)
            - deterministic and robust error handling
            """
        ).strip()
        user = dedent(
            f"""
            Create tool named '{missing_tool}' for this context:
            {task_context}
            """
        ).strip()
        source = self.llm.complete(system=system, user=user, max_tokens=1600)
        source = source.strip().removeprefix("```python").removesuffix("```").strip()
        path = self.loader.install_tool_code(missing_tool, source)
        LOGGER.info("Self-improvement success: %s", path)
        return str(path)
