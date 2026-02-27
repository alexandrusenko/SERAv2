from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, PositiveInt


class RuntimeConfig(BaseModel):
    model_path: Path
    n_ctx: PositiveInt = 32768
    n_gpu_layers: int = -1
    n_batch: PositiveInt = 1024
    threads: PositiveInt = 8
    temperature: float = 0.2
    top_p: float = 0.95
    repeat_penalty: float = 1.05
    verbose: bool = False


class MemoryConfig(BaseModel):
    db_path: Path = Path("data/memory.sqlite3")
    max_results: PositiveInt = 10


class SafetyConfig(BaseModel):
    allow_shell: bool = False
    allow_network: bool = True
    allow_python_execution: bool = False
    working_dir: Path = Path("workspace")


class AgentConfig(BaseModel):
    runtime: RuntimeConfig
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    planner_max_steps: PositiveInt = 12
    improve_after_failures: PositiveInt = 2

    @classmethod
    def from_yaml(cls, path: Path) -> "AgentConfig":
        import yaml

        content = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(content)
