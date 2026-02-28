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
            You generate LangChain tools.
            Output only python code.
            Requirements:
            - create one @tool function named exactly as requested
            - define build_tool() returning that decorated tool object
            - no dangerous imports (os, subprocess, ctypes, socket, shutil)
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
