from __future__ import annotations

import ast
from pathlib import Path

from sera_agent.config.models import SafetyConfig


class SafetyError(RuntimeError):
    pass


class SafetyPolicy:
    def __init__(self, config: SafetyConfig) -> None:
        self.config = config
        self.config.working_dir.mkdir(parents=True, exist_ok=True)

    def assert_shell_allowed(self) -> None:
        if not self.config.allow_shell:
            raise SafetyError("Shell execution is disabled by policy")

    def assert_network_allowed(self) -> None:
        if not self.config.allow_network:
            raise SafetyError("Network access is disabled by policy")

    def validate_plugin_code(self, code: str) -> None:
        tree = ast.parse(code)
        blocked_imports = {"os", "subprocess", "ctypes", "socket", "shutil"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in blocked_imports:
                        raise SafetyError(f"Blocked import in plugin: {alias.name}")
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in blocked_imports:
                    raise SafetyError(f"Blocked import in plugin: {node.module}")

    def safe_path(self, requested: str) -> Path:
        candidate = (self.config.working_dir / requested).resolve()
        base = self.config.working_dir.resolve()
        if not str(candidate).startswith(str(base)):
            raise SafetyError("Path traversal detected")
        return candidate
