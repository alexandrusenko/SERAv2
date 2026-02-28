from __future__ import annotations

import json
import logging
import platform
from abc import ABC, abstractmethod
from typing import Any
from urllib import error as urlerror
from urllib import request

from tenacity import Retrying, stop_after_attempt, wait_fixed

from sera_agent.config.models import RuntimeConfig

LOGGER = logging.getLogger(__name__)


class _Backend(ABC):
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int) -> str:
        pass


class _LlamaCppBackend(_Backend):
    def __init__(self, config: RuntimeConfig) -> None:
        super().__init__(config)
        self._llm: Any | None = None

    def _validate_model_file(self) -> None:
        if self.config.model_path is None:
            raise RuntimeError("runtime.model_path is required for backend=llama_cpp")
        model_path = self.config.model_path
        if not model_path.exists():
            raise RuntimeError(
                f"Model file does not exist: {model_path}. "
                "Update runtime.model_path in config.yaml to a valid GGUF file."
            )
        if not model_path.is_file():
            raise RuntimeError(f"Model path is not a file: {model_path}")
        with model_path.open("rb") as fp:
            magic = fp.read(4)
        if magic != b"GGUF":
            raise RuntimeError(
                f"Model file is not a valid GGUF (bad magic header): {model_path}. "
                "Verify that runtime.model_path points to a real .gguf model file."
            )

    def load(self) -> None:
        self._validate_model_file()
        assert self.config.model_path is not None
        LOGGER.info("Loading GGUF model (llama_cpp): %s", self.config.model_path)

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            msg = "llama-cpp-python is required. Install with CUDA-enabled wheel on Windows."
            raise RuntimeError(msg) from exc

        try:
            self._llm = Llama(
                model_path=str(self.config.model_path),
                n_ctx=self.config.n_ctx,
                n_gpu_layers=self.config.n_gpu_layers,
                n_batch=self.config.n_batch,
                n_threads=self.config.threads,
                verbose=self.config.verbose,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to initialize llama-cpp model. "
                f"Path: {self.config.model_path}. "
                f"Python: {platform.python_version()}. "
                "Possible causes: unsupported GGUF for installed llama-cpp, corrupted file, "
                "insufficient RAM/VRAM, or incompatible llama-cpp build (especially on Python 3.14). "
                "Try upgrading/reinstalling llama-cpp-python and testing with a smaller known-good GGUF."
            ) from exc
        LOGGER.info("llama_cpp model loaded successfully")

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        if self._llm is None:
            self.load()
        assert self._llm is not None
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repeat_penalty=self.config.repeat_penalty,
        )
        return str(response["choices"][0]["message"]["content"])


class _LMStudioBackend(_Backend):
    def load(self) -> None:
        LOGGER.info(
            "Using LM Studio backend at %s (model=%s)",
            self.config.lmstudio_base_url,
            self.config.lmstudio_model,
        )

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        payload = {
            "model": self.config.lmstudio_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        endpoint = self.config.lmstudio_base_url.rstrip("/") + "/v1/chat/completions"
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                body = response.read().decode("utf-8")
        except urlerror.URLError as exc:
            raise RuntimeError(
                "Failed to connect to LM Studio API. "
                f"Check runtime.lmstudio_base_url and ensure local server is running: {endpoint}"
            ) from exc

        try:
            parsed = json.loads(body)
            return str(parsed["choices"][0]["message"]["content"])
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LM Studio response format: {body[:500]}") from exc


class LLMEngine:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._backend: _Backend = self._create_backend()
        self._loaded = False

    def _create_backend(self) -> _Backend:
        if self.config.backend == "llama_cpp":
            return _LlamaCppBackend(self.config)
        if self.config.backend == "lmstudio":
            return _LMStudioBackend(self.config)
        raise RuntimeError(f"Unsupported runtime backend: {self.config.backend}")

    def load(self) -> None:
        self._backend.load()
        self._loaded = True

    def _complete_with_loaded_backend(self, system: str, user: str, max_tokens: int) -> str:
        LOGGER.debug("Dispatching completion request to backend=%s", self.config.backend)
        text = self._backend.complete(system=system, user=user, max_tokens=max_tokens)
        LOGGER.debug("LLM response length: %s", len(text))
        return text

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        if not self._loaded:
            self.load()

        retrying = Retrying(wait=wait_fixed(1), stop=stop_after_attempt(2), reraise=True)
        for attempt in retrying:
            with attempt:
                return self._complete_with_loaded_backend(system=system, user=user, max_tokens=max_tokens)
        raise RuntimeError("LLM completion failed")


    def complete_with_tool_calling(
        self,
        *,
        system: str,
        user: str,
        tools_schema: list[dict[str, Any]],
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        schema_hint = (
            '{"thought":"...","tool_calls":[{"name":"tool_name","arguments":{}}],'
            '"final_output":"...","success":true}'
        )
        tools_block = json.dumps(tools_schema, ensure_ascii=False)
        composed_user = (
            f"{user}\n\nAvailable tools in JSON schema:\n{tools_block}\n\n"
            "Return strictly JSON for tool calling. "
            "If no tool is required return empty tool_calls and fill final_output."
        )
        return self.complete_json(system=system, user=composed_user, schema_hint=schema_hint, max_tokens=max_tokens)

    def complete_json(self, system: str, user: str, schema_hint: str, max_tokens: int = 1024) -> dict[str, Any]:
        prompt = (
            f"{user}\n\nReturn strict JSON only. Schema hint:\n{schema_hint}\n"
            "Do not include markdown fences or commentary."
        )
        raw = self.complete(system=system, user=prompt, max_tokens=max_tokens)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            LOGGER.error("Failed to parse JSON output: %s", raw)
            raise ValueError("Model did not return valid JSON") from exc
