from __future__ import annotations

import json
from pathlib import Path

import pytest

from sera_agent.config.models import RuntimeConfig
from sera_agent.runtime.llm_engine import LLMEngine


class _FailOnceBackend:
    def __init__(self) -> None:
        self.calls = 0

    def load(self) -> None:
        return None

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        del system, user, max_tokens
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary backend failure")
        return '{"ok": true}'


class _SpyLoadEngine(LLMEngine):
    def __init__(self, config: RuntimeConfig) -> None:
        super().__init__(config)
        self.load_calls = 0
        self.fake = _FailOnceBackend()
        self._backend = self.fake

    def load(self) -> None:
        self.load_calls += 1
        self._loaded = True


def test_load_fails_fast_when_model_file_missing(tmp_path: Path) -> None:
    config = RuntimeConfig(backend="llama_cpp", model_path=tmp_path / "missing.gguf")
    engine = LLMEngine(config)

    with pytest.raises(RuntimeError, match="Model file does not exist"):
        engine.load()


def test_complete_retries_generation_without_reloading_model(tmp_path: Path) -> None:
    config = RuntimeConfig(backend="llama_cpp", model_path=tmp_path / "model.gguf")
    engine = _SpyLoadEngine(config)

    raw = engine.complete(system="s", user="u")

    assert raw == '{"ok": true}'
    assert engine.load_calls == 1


def test_load_fails_when_file_is_not_gguf(tmp_path: Path) -> None:
    fake = tmp_path / "not-model.gguf"
    fake.write_bytes(b"NOTG" + b"x" * 32)
    config = RuntimeConfig(backend="llama_cpp", model_path=fake)
    engine = LLMEngine(config)

    with pytest.raises(RuntimeError, match="bad magic header"):
        engine.load()


def test_lmstudio_backend_calls_openai_compatible_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    config = RuntimeConfig(backend="lmstudio")
    engine = LLMEngine(config)

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb
            return None

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "hi from lmstudio"}}]}).encode("utf-8")

    def _fake_urlopen(req: object, timeout: int) -> _Response:
        del timeout
        assert getattr(req, "full_url", "").endswith("/v1/chat/completions")
        return _Response()

    monkeypatch.setattr("sera_agent.runtime.llm_engine.request.urlopen", _fake_urlopen)

    assert engine.complete(system="s", user="u") == "hi from lmstudio"
