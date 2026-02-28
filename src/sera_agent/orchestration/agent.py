from __future__ import annotations

import json
import logging
import re
from textwrap import dedent
from typing import Any, Callable

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from sera_agent.config.models import AgentConfig
from sera_agent.core.types import MemoryItem, StepResult, ToolCall
from sera_agent.memory.store import MemoryStore
from sera_agent.runtime.llm_engine import LLMEngine
from sera_agent.self_improve.improver import SelfImprover
from sera_agent.tools.base import ToolRegistry

LOGGER = logging.getLogger(__name__)


class SeraAgent:
    SUPERVISOR_PROMPT = dedent(
        """
        Ты — SERA, главный модуль Supervisor в многоагентной архитектуре.
        Твоя роль: принимать управленческое решение по входной задаче.

        Твои режимы:
        - direct: ответить сразу без планирования и инструментов.
        - task: запустить полный контур (Planner -> Actor -> Critic).
        - idle: перейти в режим мета-познания (самообучение в простое).

        Выбирай режим строго по смыслу запроса.
        Возвращай только JSON.
        """
    ).strip()

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

    def _invoke_chat_completion(self, payload: dict[str, str], max_tokens: int = 1024) -> str:
        system = payload.get("system", "")
        user = payload.get("user", "")
        return self.llm.complete(system=system, user=user, max_tokens=max_tokens)

    def _invoke_json_chain(
        self,
        *,
        system: str,
        user: str,
        schema_hint: str,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        if hasattr(self.llm, "complete_json"):
            payload = self.llm.complete_json(system=system, user=user, schema_hint=schema_hint, max_tokens=max_tokens)
            return dict(payload)

        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system}"),
                (
                    "user",
                    "{user}\n\nВерни строго JSON по схеме:\n{schema_hint}\n"
                    "Без markdown, пояснений и дополнительных полей.",
                ),
            ]
        )
        llm_runnable = RunnableLambda(
            lambda messages: self._invoke_chat_completion(
                {
                    "system": str(messages[0].content),
                    "user": str(messages[-1].content),
                },
                max_tokens=max_tokens,
            )
        )
        chain = prompt | llm_runnable | parser
        return dict(chain.invoke({"system": system, "user": user, "schema_hint": schema_hint}))

    def _invoke_text_chain(self, *, system: str, user: str, max_tokens: int = 600) -> str:
        if hasattr(self.llm, "complete"):
            prompt = ChatPromptTemplate.from_messages([("system", "{system}"), ("user", "{user}")])
            llm_runnable = RunnableLambda(
                lambda messages: self._invoke_chat_completion(
                    {
                        "system": str(messages[0].content),
                        "user": str(messages[-1].content),
                    },
                    max_tokens=max_tokens,
                )
            )
            chain = prompt | llm_runnable | StrOutputParser()
            return str(chain.invoke({"system": system, "user": user})).strip()

        fallback = self._invoke_json_chain(
            system=system,
            user=user,
            schema_hint='{"final_output":"..."}',
            max_tokens=max_tokens,
        )
        return str(fallback.get("final_output") or fallback.get("answer") or "").strip()

    def _supervisor_route(self, task: str) -> str:
        if not task.strip():
            return "idle"
        if self._is_simple_question(task):
            return "direct"

        if not hasattr(self.llm, "complete"):
            return "task"

        schema_hint = '{"route":"direct|task|idle","reason":"краткое объяснение"}'
        user = dedent(
            f"""
            Входная задача пользователя:
            {task}

            Выбери режим работы Supervisor.
            """
        ).strip()
        try:
            payload = self._invoke_json_chain(
                system=self.SUPERVISOR_PROMPT,
                user=user,
                schema_hint=schema_hint,
                max_tokens=300,
            )
            route = str(payload.get("route", "task")).strip().lower()
            if route in {"direct", "task", "idle"}:
                return route
            return "task"
        except Exception:  # noqa: BLE001
            LOGGER.exception("Supervisor routing via LLM failed, fallback to heuristic route")
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
        system = "Ты SERA. Отвечай кратко и по делу на языке пользователя. Не строй план и не вызывай инструменты."
        user = dedent(
            f"""
            Задача пользователя: {task}

            Дай прямой краткий ответ.
            """
        ).strip()
        return self._invoke_text_chain(system=system, user=user, max_tokens=300)

    def _summarize_final_answer(self, task: str, transcripts: list[str]) -> str:
        raw_result = "\n\n".join(transcripts).strip()
        if not raw_result:
            return "Задача завершена, но агент не получил содержательных результатов."

        if not hasattr(self.llm, "complete"):
            return raw_result

        system = (
            "Ты SERA. Подготовь один итоговый ответ для пользователя на основе выполненных шагов. "
            "Не включай внутренние логи и служебные трассировки."
        )
        user = dedent(
            f"""
            Исходная задача пользователя: {task}

            Транскрипт выполнения агента (Agent execution transcript):
            {raw_result}

            Сформируй короткий финальный ответ на языке пользователя.
            """
        ).strip()
        try:
            summarized = self._invoke_text_chain(system=system, user=user, max_tokens=600)
            return summarized or raw_result
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to synthesize final answer from transcript")
            return raw_result

    def _plan(self, task: str, event_handler: Callable[[dict[str, str]], None] | None = None) -> list[str]:
        self._emit(event_handler, "log", "Поиск релевантной памяти...")
        context = self.memory.search(task, limit=self.config.memory.max_results)
        self._emit(event_handler, "log", f"Найдено записей памяти: {len(context)}")
        context_block = "\n".join(f"[{m.role}] {m.content}" for m in context)

        system = "Ты модуль Planner в SERA. Составь устойчивый пошаговый план выполнения задачи."
        user = dedent(
            f"""
            Задача: {task}
            Доступные инструменты:
            {self.tools.descriptions()}

            Релевантная память:
            {context_block}
            """
        ).strip()
        schema_hint = '{"steps": ["step1", "step2"]}'
        payload = self._invoke_json_chain(system=system, user=user, schema_hint=schema_hint)
        steps = payload.get("steps", [])
        if not isinstance(steps, list) or not steps:
            raise ValueError("Planner produced empty plan")
        plan = [str(s) for s in steps[: self.config.planner_max_steps]]
        self._emit(event_handler, "plan", "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(plan)))
        return plan

    def _replan(self, task: str, failed_step: str, critique: str, previous_plan: list[str]) -> list[str]:
        system = "Ты модуль Planner в SERA. Перепланируй задачу после провала шага, не меняя цель."
        user = dedent(
            f"""
            Глобальная задача: {task}
            Предыдущий план:
            {json.dumps(previous_plan, ensure_ascii=False)}
            Проваленный шаг: {failed_step}
            Комментарий Critic: {critique}

            Верни новый рабочий план.
            """
        ).strip()
        schema_hint = '{"steps": ["step1", "step2"]}'
        payload = self._invoke_json_chain(system=system, user=user, schema_hint=schema_hint)
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
        system = "Ты модуль Critic в SERA. Объясни причину ошибки простыми словами и предложи исправление."
        user = dedent(
            f"""
            Задача: {task}
            Проваленный шаг: {step}
            Детали ошибки:
            {output}

            Текущее состояние scratchpad:
            {scratchpad}

            Верни диагностику и адаптацию.
            """
        ).strip()
        schema_hint = '{"critique":"...","next_steps":["..."]}'
        payload = self._invoke_json_chain(system=system, user=user, schema_hint=schema_hint, max_tokens=1200)
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

    def _invoke_actor_tool_call(
        self,
        *,
        task: str,
        step: str,
        observations: list[str],
    ) -> dict[str, Any]:
        system = (
            "Ты модуль Actor в SERA (LangChain tool-calling style). "
            "Реши, какие инструменты вызвать на текущей итерации, а когда достаточно — верни final_output."
        )
        user = dedent(
            f"""
            Глобальная задача: {task}
            Текущий шаг: {step}

            История наблюдений/результатов инструментов:
            {chr(10).join(observations) if observations else 'нет'}

            Доступные инструменты:
            {self.tools.descriptions()}
            """
        ).strip()
        if hasattr(self.llm, "complete_with_tool_calling"):
            specs = [
                {"name": spec.name, "description": spec.description, "parameters": spec.args_schema}
                for spec in self.tools.specs()
            ]
            return dict(
                self.llm.complete_with_tool_calling(
                    system=system,
                    user=user,
                    tools_schema=specs,
                    max_tokens=1500,
                )
            )

        schema_hint = (
            '{"thought":"...","tool_calls":[{"name":"read_file","arguments":{"path":"a.txt"}}],'
            '"final_output":"...","success":true}'
        )
        return self._invoke_json_chain(system=system, user=user, schema_hint=schema_hint, max_tokens=1500)

    def _execute_step(
        self,
        task: str,
        step: str,
        event_handler: Callable[[dict[str, str]], None] | None = None,
    ) -> StepResult:
        self._emit(event_handler, "log", f"Подготовка шага: {step}")
        tool_calls: list[ToolCall] = []
        observations: list[str] = []
        final_output = ""
        success = False

        for iteration in range(1, 4):
            payload = self._invoke_actor_tool_call(task=task, step=step, observations=observations)
            calls_raw = payload.get("tool_calls", [])
            if not isinstance(calls_raw, list):
                calls_raw = []
            self._emit(event_handler, "log", f"Tool-calling итерация {iteration}, вызовов: {len(calls_raw)}")

            if not calls_raw:
                final_output = str(payload.get("final_output", "")).strip()
                success = bool(payload.get("success", bool(final_output)))
                break

            for c in calls_raw:
                name = str(c.get("name", "")).strip()
                args_obj = c.get("arguments", {})
                args = args_obj if isinstance(args_obj, dict) else {}
                tool_calls.append(ToolCall(name=name, arguments=args))
                self._emit(event_handler, "log", f"Вызов инструмента: {name}({json.dumps(args, ensure_ascii=False)})")

                if not self.tools.has(name):
                    self.improver.attempt_create_missing_tool(
                        missing_tool=name,
                        task_context=(
                            "LangChain tool-calling runtime. "
                            f"Task: {task}; Step: {step}; Observations: {chr(10).join(observations[-4:])}"
                        ),
                    )
                    self._emit(event_handler, "log", f"Инструмент {name} отсутствовал, создан динамически")

                if not self.tools.has(name):
                    observations.append(f"{name}: success=False output=Tool is unavailable")
                    continue

                tool = self.tools.get(name)
                try:
                    result = tool.run(args)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Tool call failed: %s", name)
                    observations.append(f"{name}: success=False output=Tool error: {exc}")
                    continue

                self._emit(event_handler, "log", f"Результат {name}: success={result.success}")
                observations.append(f"{name}: success={result.success} output={result.output[:700]}")

            final_output = str(payload.get("final_output", "")).strip()
            success = bool(payload.get("success", False))

        combined = "\n".join([*observations, final_output]).strip()
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
        system = "Ты модуль Metacognition в SERA. Сгенерируй одну практичную задачу самообучения."
        user = dedent(
            f"""
            Контекст недавних ошибок:
            {failure_context}

            Предложи микро-цикл обучения.
            """
        ).strip()
        schema_hint = '{"learning_goal":"...","experiment_plan":["..."],"self_test":"..."}'
        payload = self._invoke_json_chain(system=system, user=user, schema_hint=schema_hint, max_tokens=1200)
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
            self._emit(event_handler, "log", "Supervisor выбрал direct: отвечаю без планирования")
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

        final = self._summarize_final_answer(task=task, transcripts=transcripts)
        self.memory.add(MemoryItem(role="assistant", content=f"FINAL:\n{final}"))
        self._emit(event_handler, "final", final)
        return final
