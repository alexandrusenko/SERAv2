from __future__ import annotations

import json
import logging
import re
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
        self.scratchpad_path = self.config.safety.working_dir / "memory.md"
        self.idle_reflection_threshold_seconds = 600

    def _supervisor_route(self, task: str) -> str:
        if not task.strip():
            return "idle"
        if self._is_simple_question(task):
            return "direct"
        return "task"

    def _emit(
        self,
        event_handler: Callable[[dict[str, str]], None] | None,
        kind: str,
        message: str,
    ) -> None:
        if event_handler is None:
            return
        event_handler({"kind": kind, "message": message})

    def _is_simple_question(self, task: str) -> bool:
        normalized = task.strip().lower()
        if not normalized:
            return True
        quick_patterns = (
            "привет",
            "hello",
            "hi",
            "как дела",
            "кто ты",
            "что ты умеешь",
            "доброе утро",
            "добрый день",
        )
        return any(normalized == pattern for pattern in quick_patterns)

    def _answer_directly(self, task: str) -> str:
        system = "You are a concise assistant. Answer directly and do not plan or call any tools."
        user = dedent(
            f"""
            User task: {task}

            Provide a short direct answer in the same language as the user.
            """
        ).strip()
        return self.llm.complete(system=system, user=user, max_tokens=300).strip()

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

    def _replan(self, task: str, failed_step: str, critique: str, previous_plan: list[str]) -> list[str]:
        system = "You are a resilient planner. Replan after failures while keeping objective unchanged."
        user = dedent(
            f"""
            Global task: {task}
            Previous plan:
            {json.dumps(previous_plan, ensure_ascii=False)}
            Failed step: {failed_step}
            Critic feedback: {critique}

            Return JSON with field "steps": a new robust plan.
            """
        ).strip()
        schema_hint = '{"steps": ["step1", "step2"]}'
        payload = self.llm.complete_json(system=system, user=user, schema_hint=schema_hint)
        steps = payload.get("steps", [])
        if not isinstance(steps, list) or not steps:
            return previous_plan
        return [str(s) for s in steps[: self.config.planner_max_steps]]

    def _read_scratchpad(self) -> str:
        if not self.scratchpad_path.exists():
            return ""
        return self.scratchpad_path.read_text(encoding="utf-8")

    def _write_scratchpad(self, content: str) -> None:
        self.scratchpad_path.parent.mkdir(parents=True, exist_ok=True)
        self.scratchpad_path.write_text(content, encoding="utf-8")

    def _initialize_scratchpad(self, task: str, plan: list[str]) -> None:
        tasks = "\n".join(f"- [ ] {step}" for step in plan)
        content = dedent(
            f"""
            # Scratchpad
            Task: {task}

            ## Dynamic Plan
            {tasks}

            ## Execution Log

            ## Reflections
            """
        ).strip()
        self._write_scratchpad(content)

    def _append_scratchpad_section(self, title: str, body: str) -> None:
        current = self._read_scratchpad().rstrip()
        addition = f"\n\n### {title}\n{body.strip()}\n"
        self._write_scratchpad(f"{current}{addition}")

    def _mark_step(self, step: str, done: bool) -> None:
        marker_from = f"- [ ] {step}"
        marker_to = f"- [{'x' if done else '!'}] {step}"
        scratchpad = self._read_scratchpad()
        if marker_from in scratchpad:
            self._write_scratchpad(scratchpad.replace(marker_from, marker_to, 1))

    def _reflect_on_failure(self, task: str, step: str, output: str) -> tuple[str, list[str]]:
        scratchpad = self._read_scratchpad()
        system = "You are a strict critic. Explain why the attempt failed and how to adapt the next action."
        user = dedent(
            f"""
            Task: {task}
            Failed step: {step}
            Failure details:
            {output}

            Scratchpad state:
            {scratchpad}

            Return JSON with fields:
            - critique: string
            - next_steps: list of short strings
            """
        ).strip()
        schema_hint = '{"critique":"...","next_steps":["..."]}'
        payload = self.llm.complete_json(system=system, user=user, schema_hint=schema_hint, max_tokens=1200)
        critique = str(payload.get("critique") or payload.get("thought") or payload.get("final_output") or "No critique")
        next_steps_raw = payload.get("next_steps", [])
        next_steps = [str(item) for item in next_steps_raw if str(item).strip()]
        if next_steps:
            self._append_scratchpad_section("Adaptive updates", "\n".join(f"- [ ] {item}" for item in next_steps))
        self._append_scratchpad_section("Reflection", critique)
        return critique, next_steps

    def _adapt_plan(self, plan: list[str], current_index: int, next_steps: list[str]) -> list[str]:
        if not next_steps:
            return plan
        updated = [*plan[:current_index], *next_steps, *plan[current_index + 1 :]]
        return updated[: self.config.planner_max_steps]

    def _execute_step(
        self,
        task: str,
        step: str,
        event_handler: Callable[[dict[str, str]], None] | None = None,
    ) -> StepResult:
        self._emit(event_handler, "log", f"Подготовка шага: {step}")
        system = "You are the Actor module. Execute one step, choose tools, and report verifiable result. For casual greetings or basic knowledge, avoid network tools unless the user explicitly asks to search online or current web data is required."
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
        self._append_scratchpad_section("Execution Log", f"Step: {step}\n\n{combined}")
        self._emit(event_handler, "step", f"{step}\n\n{combined}")
        return StepResult(step=step, success=success, output=combined, tool_calls=tool_calls)

    def _extract_missing_module(self, output: str) -> str | None:
        patterns = [
            r"No module named ['\"]([^'\"]+)['\"]",
            r"ModuleNotFoundError:\s*([^\s]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return match.group(1).strip()
        return None

    def _try_auto_install_dependency(
        self,
        task: str,
        step: str,
        output: str,
        event_handler: Callable[[dict[str, str]], None] | None = None,
    ) -> bool:
        missing_module = self._extract_missing_module(output)
        if not missing_module:
            return False
        if not self.tools.has("shell"):
            return False

        install_cmd = f"python -m pip install {missing_module}"
        self._emit(event_handler, "log", f"Автоисправление: пытаюсь установить зависимость {missing_module}")
        result = self.tools.get("shell").run({"command": install_cmd})
        self.memory.add(
            MemoryItem(
                role="assistant",
                content=f"AUTO_FIX for {task} / {step}: {install_cmd}\nsuccess={result.success}\n{result.output}",
            )
        )
        self._append_scratchpad_section(
            "Auto-fix",
            f"Command: {install_cmd}\nSuccess: {result.success}\nOutput:\n{result.output[:1500]}",
        )
        return result.success

    def run_idle_cycle(self, event_handler: Callable[[dict[str, str]], None] | None = None) -> str:
        self._emit(event_handler, "log", "Idle mode: запуск мета-познания")
        failures = self.memory.search("error failure exception", limit=6)
        failure_context = "\n".join(f"[{item.role}] {item.content}" for item in failures)
        system = "You are the Metacognition module. Generate one practical self-improvement task."
        user = dedent(
            f"""
            Recent failure context:
            {failure_context}

            Return JSON with fields:
            - learning_goal: string
            - experiment_plan: list of strings
            - self_test: string (a tiny verifiable check)
            """
        ).strip()
        schema_hint = '{"learning_goal":"...","experiment_plan":["..."],"self_test":"..."}'
        payload = self.llm.complete_json(system=system, user=user, schema_hint=schema_hint, max_tokens=1200)
        goal = str(payload.get("learning_goal") or "Уточнить пробелы в навыках")
        plan = payload.get("experiment_plan") or []
        steps = [str(item) for item in plan if str(item).strip()]
        self_test = str(payload.get("self_test") or "assert True")

        reflection = dedent(
            f"""
            Idle Reflection
            Goal: {goal}
            Plan:
            {chr(10).join(f"- {item}" for item in steps) if steps else "- Сформировать новый микро-эксперимент"}
            Self-test: {self_test}
            """
        ).strip()
        self.memory.add(MemoryItem(role="assistant", content=reflection))
        self._append_scratchpad_section("Metacognition", reflection)
        self._emit(event_handler, "reflection", reflection)
        return reflection

    def run(self, task: str, event_handler: Callable[[dict[str, str]], None] | None = None) -> str:
        LOGGER.info("Agent run started: %s", task)
        self._emit(event_handler, "log", f"Старт задачи: {task}")
        self.memory.add(MemoryItem(role="user", content=task))

        route = self._supervisor_route(task)
        if route == "idle":
            idle_result = self.run_idle_cycle(event_handler=event_handler)
            self._emit(event_handler, "final", idle_result)
            return idle_result

        if route == "direct":
            self._emit(event_handler, "log", "Простой запрос: отвечаю сразу без планирования")
            direct = self._answer_directly(task)
            self.memory.add(MemoryItem(role="assistant", content=direct))
            self._emit(event_handler, "final", direct)
            return direct

        plan = self._plan(task, event_handler=event_handler)
        self._initialize_scratchpad(task=task, plan=plan)
        LOGGER.info("Plan generated with %d steps", len(plan))
        self._emit(event_handler, "log", f"План готов, шагов: {len(plan)}")

        failures = 0
        transcripts: list[str] = []

        idx = 0
        max_iterations = self.config.planner_max_steps * 2
        iterations = 0

        while idx < len(plan) and iterations < max_iterations:
            iterations += 1
            step = plan[idx]
            LOGGER.info("Executing step %d/%d: %s", idx + 1, len(plan), step)
            self._emit(event_handler, "log", f"Выполнение шага {idx + 1}/{len(plan)}")
            result = self._execute_step(task=task, step=step, event_handler=event_handler)
            transcripts.append(f"STEP {idx + 1}: {result.step}\n{result.output}")
            self.memory.add(MemoryItem(role="assistant", content=result.output))

            if not result.success:
                failures += 1
                self._mark_step(step, done=False)
                self._append_scratchpad_section("Execution failure", f"Step: {step}\n\n{result.output}")
                LOGGER.warning("Step failed (count=%d)", failures)
                self._emit(event_handler, "log", f"Шаг завершился неуспешно, подряд ошибок: {failures}")
                critique, critic_steps = self._reflect_on_failure(task=task, step=step, output=result.output)
                self._emit(event_handler, "reflection", critique)

                if self._try_auto_install_dependency(task=task, step=step, output=result.output, event_handler=event_handler):
                    self._emit(event_handler, "log", "Автоисправление зависимости успешно, повтор шага")
                    continue

                adaptive_steps = []
                scratchpad = self._read_scratchpad()
                for line in scratchpad.splitlines():
                    if line.startswith("- [ ] "):
                        candidate = line.replace("- [ ] ", "", 1).strip()
                        if candidate and candidate not in plan[idx + 1 :]:
                            adaptive_steps.append(candidate)
                adaptive_steps = [*critic_steps, *adaptive_steps]
                plan = self._adapt_plan(plan=plan, current_index=idx, next_steps=adaptive_steps[:2])
                plan = self._replan(task=task, failed_step=step, critique=critique, previous_plan=plan)
                self._emit(event_handler, "plan", "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan)))
                if failures >= self.config.improve_after_failures:
                    self.improver.attempt_create_missing_tool(
                        missing_tool="task_specific_helper",
                        task_context=f"Repeated failures for task: {task}. Recent step: {step}",
                    )
                    self._emit(event_handler, "log", "Запущено улучшение task_specific_helper после повторных ошибок")
                if adaptive_steps:
                    continue
                idx += 1
            else:
                failures = 0
                self._mark_step(step, done=True)
                self._append_scratchpad_section("Execution success", f"Step: {step}\n\n{result.output}")
                idx += 1

        if iterations >= max_iterations:
            LOGGER.warning("Execution stopped by iteration guard")
            self._append_scratchpad_section("Guard", "Iteration limit reached, stopping execution")

        final = "\n\n".join(transcripts)
        self.memory.add(MemoryItem(role="assistant", content=f"FINAL:\n{final}"))
        self._emit(event_handler, "final", final)
        return final
