from __future__ import annotations

import json
import logging
import subprocess
from html.parser import HTMLParser

from langchain_core.tools import BaseTool, tool

from sera_agent.safety.policy import SafetyPolicy

LOGGER = logging.getLogger(__name__)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return "\n".join(self.parts)


def build_read_file_tool(safety: SafetyPolicy) -> BaseTool:
    @tool("read_file")
    def read_file(path: str) -> str:
        """Read a UTF-8 text file from the safe working directory."""
        safe = safety.safe_path(path)
        if not safe.exists():
            return f"File not found: {path}"
        return safe.read_text(encoding="utf-8")

    return read_file


def build_write_file_tool(safety: SafetyPolicy) -> BaseTool:
    @tool("write_file")
    def write_file(path: str, content: str) -> str:
        """Write UTF-8 text to a file in the safe working directory."""
        safe = safety.safe_path(path)
        safe.parent.mkdir(parents=True, exist_ok=True)
        safe.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {path}"

    return write_file


def build_shell_tool(safety: SafetyPolicy) -> BaseTool:
    @tool("shell")
    def shell(command: str) -> str:
        """Execute a shell command and return JSON with returncode/stdout/stderr."""
        safety.assert_shell_allowed()
        LOGGER.info("ShellTool command=%s", command)
        completed = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return json.dumps(
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
            ensure_ascii=False,
        )

    return shell


def build_web_search_tool(safety: SafetyPolicy) -> BaseTool:
    @tool("web_search")
    def web_search(query: str, max_results: int = 5) -> str:
        """Search the web using DDGS and return compact JSON results."""
        safety.assert_network_allowed()
        if not query.strip():
            return "Missing query"

        from ddgs import DDGS

        with DDGS() as ddgs:
            rows = list(ddgs.text(query, max_results=max(1, min(max_results, 10))))
        compact = [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("href", "")),
                "snippet": str(item.get("body", "")),
            }
            for item in rows
        ]
        return json.dumps(compact, ensure_ascii=False)

    return web_search


def build_fetch_url_tool(safety: SafetyPolicy) -> BaseTool:
    @tool("fetch_url")
    def fetch_url(url: str) -> str:
        """Fetch URL content and return readable text."""
        safety.assert_network_allowed()
        import urllib.request

        with urllib.request.urlopen(url, timeout=20) as response:  # noqa: S310
            body = response.read().decode("utf-8", errors="replace")
        parser = _TextExtractor()
        parser.feed(body)
        return (parser.text() or body)[:10000]

    return fetch_url
