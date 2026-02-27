from __future__ import annotations

import json
import logging
from typing import Any

from tenacity import retry, stop_after_attempt, wait_fixed

from sera_agent.config.models import RuntimeConfig

LOGGER = logging.getLogger(__name__)


class LLMEngine:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._llm: Any | None = None

    def load(self) -> None:
        LOGGER.info("Loading GGUF model: %s", self.config.model_path)
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            msg = "llama-cpp-python is required. Install with CUDA-enabled wheel on Windows."
            raise RuntimeError(msg) from exc

        self._llm = Llama(
            model_path=str(self.config.model_path),
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
            n_batch=self.config.n_batch,
            n_threads=self.config.threads,
            verbose=self.config.verbose,
        )
        LOGGER.info("Model loaded successfully")

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(2), reraise=True)
    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        if self._llm is None:
            self.load()
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
