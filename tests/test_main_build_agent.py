from __future__ import annotations

from pathlib import Path

from sera_agent.main import build_agent


def _write_config(path: Path, allow_shell: bool) -> None:
    path.write_text(
        "\n".join(
            [
                "runtime:",
                '  model_path: "model.gguf"',
                "memory:",
                '  db_path: "data/memory.sqlite3"',
                "safety:",
                f"  allow_shell: {'true' if allow_shell else 'false'}",
                "  allow_network: true",
                "  allow_python_execution: false",
                '  working_dir: "workspace"',
            ]
        ),
        encoding="utf-8",
    )


def test_build_agent_does_not_register_shell_when_disabled(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    _write_config(cfg, allow_shell=False)

    agent = build_agent(cfg)

    assert not agent.tools.has("shell")


def test_build_agent_registers_shell_when_enabled(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    _write_config(cfg, allow_shell=True)

    agent = build_agent(cfg)

    assert agent.tools.has("shell")
