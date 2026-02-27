from __future__ import annotations

from pathlib import Path

import pytest

from sera_agent.config.models import RuntimeConfig
from sera_agent.runtime.llm_engine import LLMEngine


class _FailOnceLLM:
    def __init__(self) -> None:
        self.calls = 0

    def create_chat_completion(self, **_: object) -> dict[str, object]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary backend failure")
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}


class _SpyLoadEngine(LLMEngine):
    def __init__(self, config: RuntimeConfig) -> None:
        super().__init__(config)
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1
        self._llm = _FailOnceLLM()


def test_load_fails_fast_when_model_file_missing(tmp_path: Path) -> None:
    config = RuntimeConfig(model_path=tmp_path / "missing.gguf")
    engine = LLMEngine(config)

    with pytest.raises(RuntimeError, match="Model file does not exist"):
        engine.load()


def test_complete_retries_generation_without_reloading_model(tmp_path: Path) -> None:
    config = RuntimeConfig(model_path=tmp_path / "model.gguf")
    engine = _SpyLoadEngine(config)

    raw = engine.complete(system="s", user="u")

    assert raw == '{"ok": true}'
    assert engine.load_calls == 1


def test_load_fails_when_file_is_not_gguf(tmp_path: Path) -> None:
    fake = tmp_path / "not-model.gguf"
    fake.write_bytes(b"NOTG" + b"x" * 32)
    config = RuntimeConfig(model_path=fake)
    engine = LLMEngine(config)

    with pytest.raises(RuntimeError, match="bad magic header"):
        engine.load()
