from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sera_agent.safety.policy import SafetyPolicy
from sera_agent.tools.base import ToolResult

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ReadFileTool:
    safety: SafetyPolicy
    name: str = "read_file"
    description: str = "Read a UTF-8 text file from the safe working directory. args: {path}"

    def run(self, arguments: dict[str, object]) -> ToolResult:
        raw_path = str(arguments.get("path", ""))
        path = self.safety.safe_path(raw_path)
        if not path.exists():
            return ToolResult(False, f"File not found: {raw_path}")
        return ToolResult(True, path.read_text(encoding="utf-8"))


@dataclass(slots=True)
class WriteFileTool:
    safety: SafetyPolicy
    name: str = "write_file"
    description: str = "Write UTF-8 text file to safe working directory. args: {path, content}"

    def run(self, arguments: dict[str, object]) -> ToolResult:
        raw_path = str(arguments.get("path", ""))
        content = str(arguments.get("content", ""))
        path = self.safety.safe_path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult(True, f"Wrote {len(content)} chars to {raw_path}")


@dataclass(slots=True)
class ShellTool:
    safety: SafetyPolicy
    name: str = "shell"
    description: str = "Execute shell command. args: {command}"

    def run(self, arguments: dict[str, object]) -> ToolResult:
        self.safety.assert_shell_allowed()
        command = str(arguments.get("command", ""))
        LOGGER.info("ShellTool command=%s", command)
        try:
            completed = subprocess.run(
                command,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except Exception as exc:
            return ToolResult(False, f"Shell error: {exc}")

        payload = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        success = completed.returncode == 0
        return ToolResult(success, json.dumps(payload, ensure_ascii=False))


@dataclass(slots=True)
class HttpGetTool:
    safety: SafetyPolicy
    name: str = "http_get"
    description: str = "GET URL and return text. args: {url}"

    def run(self, arguments: dict[str, object]) -> ToolResult:
        self.safety.assert_network_allowed()
        url = str(arguments.get("url", ""))
        try:
            import urllib.request

            with urllib.request.urlopen(url, timeout=20) as response:  # noqa: S310
                body = response.read().decode("utf-8", errors="replace")
            return ToolResult(True, body[:10000])
        except Exception as exc:
            return ToolResult(False, f"HTTP error: {exc}")
