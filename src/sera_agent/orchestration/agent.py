from __future__ import annotations

import json
import logging
from textwrap import dedent
from typing import Callable

from sera_agent.config.models import AgentConfig
from sera_agent.core.types import MemoryItem, StepResult, ToolCall
from sera_agent.memory.store import MemoryStore
from sera_agent.runtime.llm_engine import LLMEngine
from sera_agent.self_improve.improver import SelfImprover
from sera_agent.tools.base import ToolRegistry

LOGGER = logging.getLogger(__name__)


class SeraAgent:
    def __init__(
        self,
        config: AgentConfig,
        llm: LLMEngine,
        memory: MemoryStore,
        tools: ToolRegistry,
        improver: SelfImprover,
    ) -> None:
        self.config = config
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.improver = improver

    def _emit(
        self,
        event_handler: Callable[[dict[str, str]], None] | None,
        kind: str,
        message: str,
    ) -> None:
        if event_handler is None:
            return
        event_handler({"kind": kind, "message": message})

    def _plan(self, task: str, event_handler: Callable[[dict[str, str]], None] | None = None) -> list[str]:
        self._emit(event_handler, "log", "Поиск релевантной памяти...")
        context = self.memory.search(task, limit=self.config.memory.max_results)
        self._emit(event_handler, "log", f"Найдено записей памяти: {len(context)}")
        context_block = "\n".join(f"[{m.role}] {m.content}" for m in context)

        system = "You are a deliberate planner. Create robust step-by-step plans."
        user = dedent(
            f"""
            Task: {task}
            Available tools:
            {self.tools.descriptions()}

            Relevant memory:
            {context_block}
            """
        ).strip()
        schema_hint = '{"steps": ["step1", "step2"]}'
        payload = self.llm.complete_json(system=system, user=user, schema_hint=schema_hint)
        steps = payload.get("steps", [])
        if not isinstance(steps, list) or not steps:
            raise ValueError("Planner produced empty plan")
        plan = [str(s) for s in steps[: self.config.planner_max_steps]]
        self._emit(event_handler, "plan", "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(plan)))
        return plan

    def _execute_step(
        self,
        task: str,
        step: str,
        event_handler: Callable[[dict[str, str]], None] | None = None,
    ) -> StepResult:
        self._emit(event_handler, "log", f"Подготовка шага: {step}")
        system = "You execute one step and decide tool calls when needed."
        user = dedent(
            f"""
            Global task: {task}
            Current step: {step}
            Available tools:
            {self.tools.descriptions()}

            Return JSON with fields:
            - thought: string
            - tool_calls: list of {{name: string, arguments: object}}
            - final_output: string
            - success: boolean
            """
        ).strip()
        schema_hint = (
            '{"thought":"...","tool_calls":[{"name":"read_file","arguments":{"path":"a.txt"}}],'
            '"final_output":"...","success":true}'
        )
        payload = self.llm.complete_json(system=system, user=user, schema_hint=schema_hint, max_tokens=1500)
        calls_raw = payload.get("tool_calls", [])
        tool_calls: list[ToolCall] = []
        outputs: list[str] = []
        self._emit(event_handler, "log", f"Инструментов для вызова: {len(calls_raw)}")

        for c in calls_raw:
            name = str(c.get("name", ""))
            args_obj = c.get("arguments", {})
            args = args_obj if isinstance(args_obj, dict) else {}
            tool_calls.append(ToolCall(name=name, arguments=args))
            self._emit(event_handler, "log", f"Вызов инструмента: {name}({json.dumps(args, ensure_ascii=False)})")
            if not self.tools.has(name):
                self.improver.attempt_create_missing_tool(missing_tool=name, task_context=f"Task: {task}; Step: {step}")
                self._emit(event_handler, "log", f"Инструмент {name} отсутствовал, запущено самоулучшение")
                if not self.tools.has(name):
                    outputs.append(f"{name}: success=False output=Tool is unavailable")
                    continue

            tool = self.tools.get(name)
            try:
                result = tool.run(args)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Tool call failed: %s", name)
                error_output = f"Tool error: {exc}"
                outputs.append(f"{name}: success=False output={error_output[:500]}")
                continue

            self._emit(event_handler, "log", f"Результат {name}: success={result.success}")
            outputs.append(f"{name}: success={result.success} output={result.output[:500]}")

        final_output = str(payload.get("final_output", ""))
        success = bool(payload.get("success", False))
        combined = "\n".join([*outputs, final_output]).strip()
        self._emit(event_handler, "step", f"{step}\n\n{combined}")
        return StepResult(step=step, success=success, output=combined, tool_calls=tool_calls)

    def run(self, task: str, event_handler: Callable[[dict[str, str]], None] | None = None) -> str:
        LOGGER.info("Agent run started: %s", task)
        self._emit(event_handler, "log", f"Старт задачи: {task}")
        self.memory.add(MemoryItem(role="user", content=task))
        plan = self._plan(task, event_handler=event_handler)
        LOGGER.info("Plan generated with %d steps", len(plan))
        self._emit(event_handler, "log", f"План готов, шагов: {len(plan)}")

        failures = 0
        transcripts: list[str] = []

        for idx, step in enumerate(plan, start=1):
            LOGGER.info("Executing step %d/%d: %s", idx, len(plan), step)
            self._emit(event_handler, "log", f"Выполнение шага {idx}/{len(plan)}")
            result = self._execute_step(task=task, step=step, event_handler=event_handler)
            transcripts.append(f"STEP {idx}: {result.step}\n{result.output}")
            self.memory.add(MemoryItem(role="assistant", content=result.output))

            if not result.success:
                failures += 1
                LOGGER.warning("Step failed (count=%d)", failures)
                self._emit(event_handler, "log", f"Шаг завершился неуспешно, подряд ошибок: {failures}")
                if failures >= self.config.improve_after_failures:
                    self.improver.attempt_create_missing_tool(
                        missing_tool="task_specific_helper",
                        task_context=f"Repeated failures for task: {task}. Recent step: {step}",
                    )
                    self._emit(event_handler, "log", "Запущено улучшение task_specific_helper после повторных ошибок")
            else:
                failures = 0

        final = "\n\n".join(transcripts)
        self.memory.add(MemoryItem(role="assistant", content=f"FINAL:\n{final}"))
        self._emit(event_handler, "final", final)
        return final
