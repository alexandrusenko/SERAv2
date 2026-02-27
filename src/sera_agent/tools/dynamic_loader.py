from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

from sera_agent.safety.policy import SafetyPolicy
from sera_agent.tools.base import ToolRegistry

LOGGER = logging.getLogger(__name__)


class DynamicToolLoader:
    def __init__(self, tools_dir: Path, safety: SafetyPolicy, registry: ToolRegistry) -> None:
        self.tools_dir = tools_dir
        self.safety = safety
        self.registry = registry
        self.tools_dir.mkdir(parents=True, exist_ok=True)

    def install_tool_code(self, tool_name: str, source_code: str) -> Path:
        self.safety.validate_plugin_code(source_code)
        file_path = self.tools_dir / f"{tool_name}.py"
        file_path.write_text(source_code, encoding="utf-8")
        self.load_from_file(file_path)
        return file_path

    def load_from_file(self, file_path: Path) -> None:
        module_name = f"dynamic_tool_{file_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load tool module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "build_tool"):
            raise RuntimeError("Dynamic tool must define build_tool()")
        tool = module.build_tool()
        self.registry.register(tool)
        LOGGER.info("Dynamic tool loaded: %s", tool.name)
