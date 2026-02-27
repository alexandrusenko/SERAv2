from __future__ import annotations

import json
import logging
from textwrap import dedent

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

    def _plan(self, task: str) -> list[str]:
        context = self.memory.search(task, limit=self.config.memory.max_results)
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
        return [str(s) for s in steps[: self.config.planner_max_steps]]

    def _execute_step(self, task: str, step: str) -> StepResult:
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

        for c in calls_raw:
            name = str(c.get("name", ""))
            args_obj = c.get("arguments", {})
            args = args_obj if isinstance(args_obj, dict) else {}
            tool_calls.append(ToolCall(name=name, arguments=args))
            if not self.tools.has(name):
                self.improver.attempt_create_missing_tool(missing_tool=name, task_context=f"Task: {task}; Step: {step}")
            tool = self.tools.get(name)
            result = tool.run(args)
            outputs.append(f"{name}: success={result.success} output={result.output[:500]}")

        final_output = str(payload.get("final_output", ""))
        success = bool(payload.get("success", False))
        combined = "\n".join([*outputs, final_output]).strip()
        return StepResult(step=step, success=success, output=combined, tool_calls=tool_calls)

    def run(self, task: str) -> str:
        LOGGER.info("Agent run started: %s", task)
        self.memory.add(MemoryItem(role="user", content=task))
        plan = self._plan(task)
        LOGGER.info("Plan generated with %d steps", len(plan))

        failures = 0
        transcripts: list[str] = []

        for idx, step in enumerate(plan, start=1):
            LOGGER.info("Executing step %d/%d: %s", idx, len(plan), step)
            result = self._execute_step(task=task, step=step)
            transcripts.append(f"STEP {idx}: {result.step}\n{result.output}")
            self.memory.add(MemoryItem(role="assistant", content=result.output))

            if not result.success:
                failures += 1
                LOGGER.warning("Step failed (count=%d)", failures)
                if failures >= self.config.improve_after_failures:
                    self.improver.attempt_create_missing_tool(
                        missing_tool="task_specific_helper",
                        task_context=f"Repeated failures for task: {task}. Recent step: {step}",
                    )
            else:
                failures = 0

        final = "\n\n".join(transcripts)
        self.memory.add(MemoryItem(role="assistant", content=f"FINAL:\n{final}"))
        return final
