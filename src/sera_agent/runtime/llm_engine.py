from __future__ import annotations

import json
import logging
import platform
from typing import Any

from tenacity import Retrying, stop_after_attempt, wait_fixed

from sera_agent.config.models import RuntimeConfig

LOGGER = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._llm: Any | None = None

    def _validate_model_file(self) -> None:
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
        model_path = self.config.model_path
        LOGGER.info("Loading GGUF model: %s", model_path)
        self._validate_model_file()

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            msg = "llama-cpp-python is required. Install with CUDA-enabled wheel on Windows."
            raise RuntimeError(msg) from exc

        try:
            self._llm = Llama(
                model_path=str(model_path),
                n_ctx=self.config.n_ctx,
                n_gpu_layers=self.config.n_gpu_layers,
                n_batch=self.config.n_batch,
                n_threads=self.config.threads,
                verbose=self.config.verbose,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to initialize llama-cpp model. "
                f"Path: {model_path}. "
                f"Python: {platform.python_version()}. "
                "Possible causes: unsupported GGUF for installed llama-cpp, corrupted file, "
                "insufficient RAM/VRAM, or incompatible llama-cpp build (especially on Python 3.14). "
                "Try upgrading/reinstalling llama-cpp-python and testing with a smaller known-good GGUF."
            ) from exc
        LOGGER.info("Model loaded successfully")

    def _complete_with_loaded_model(self, system: str, user: str, max_tokens: int) -> str:
        assert self._llm is not None
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        LOGGER.debug("Dispatching completion request")
        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repeat_penalty=self.config.repeat_penalty,
        )
        text = response["choices"][0]["message"]["content"]
        LOGGER.debug("LLM response length: %s", len(text))
        return text

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        if self._llm is None:
            self.load()

        retrying = Retrying(wait=wait_fixed(1), stop=stop_after_attempt(2), reraise=True)
        for attempt in retrying:
            with attempt:
                return self._complete_with_loaded_model(system=system, user=user, max_tokens=max_tokens)
        raise RuntimeError("LLM completion failed")

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
