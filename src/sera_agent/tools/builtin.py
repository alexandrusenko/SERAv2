from __future__ import annotations

import json
import logging
import subprocess
from html.parser import HTMLParser
from dataclasses import dataclass

from sera_agent.safety.policy import SafetyPolicy
from sera_agent.tools.base import ToolResult, ToolSpec

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ReadFileTool:
    safety: SafetyPolicy
    name: str = "read_file"
    description: str = "Read a UTF-8 text file from the safe working directory. args: {path}"

    def schema(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            args_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        )

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

    def schema(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            args_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        )

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

    def schema(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            args_schema={"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        )

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
class WebSearchTool:
    safety: SafetyPolicy
    name: str = "web_search"
    description: str = "Search the web with DDGS and return top results. args: {query, max_results?}"

    def schema(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            args_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                "required": ["query"],
            },
        )

    def run(self, arguments: dict[str, object]) -> ToolResult:
        self.safety.assert_network_allowed()
        query = str(arguments.get("query", "")).strip()
        max_results_raw = arguments.get("max_results", 5)
        try:
            max_results = min(max(int(max_results_raw), 1), 10)
        except (TypeError, ValueError):
            max_results = 5

        if not query:
            return ToolResult(False, "Missing query")

        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                rows = list(ddgs.text(query, max_results=max_results))
            compact = [
                {
                    "title": str(item.get("title", "")),
                    "url": str(item.get("href", "")),
                    "snippet": str(item.get("body", "")),
                }
                for item in rows
            ]
            return ToolResult(True, json.dumps(compact, ensure_ascii=False))
        except Exception as exc:
            return ToolResult(False, f"Web search error: {exc}")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return "\n".join(self.parts)


@dataclass(slots=True)
class FetchUrlTool:
    safety: SafetyPolicy
    name: str = "fetch_url"
    description: str = "Fetch URL and return readable text content. args: {url}"

    def schema(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            args_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        )

    def run(self, arguments: dict[str, object]) -> ToolResult:
        self.safety.assert_network_allowed()
        url = str(arguments.get("url", ""))
        try:
            import urllib.request

            with urllib.request.urlopen(url, timeout=20) as response:  # noqa: S310
                body = response.read().decode("utf-8", errors="replace")
            parser = _TextExtractor()
            parser.feed(body)
            parsed = parser.text() or body
            return ToolResult(True, parsed[:10000])
        except Exception as exc:
            return ToolResult(False, f"HTTP error: {exc}")
