from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage


@dataclass
class _SimpleReactAgent:
    model: Any
    tools: list[Any]
    prompt: str

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        user_messages = payload.get("messages", [])
        task = ""
        if user_messages:
            last = user_messages[-1]
            task = last[1] if isinstance(last, tuple) else str(last)

        engine = getattr(self.model, "engine", None)
        tool_map = {tool.name: tool for tool in self.tools}
        messages: list[Any] = []

        if engine is not None and hasattr(engine, "complete_json"):
            response = engine.complete_json(
                system=self.prompt,
                user=task,
                schema_hint='{"tool_calls":[{"name":"...","arguments":{}}],"final_output":"...","success":true}',
                max_tokens=1024,
            )
            for call in response.get("tool_calls", []):
                name = str(call.get("name", ""))
                args = call.get("arguments", {})
                if name in tool_map:
                    try:
                        output = tool_map[name].invoke(args if isinstance(args, dict) else {})
                        messages.append(ToolMessage(content=str(output)))
                    except Exception as exc:  # noqa: BLE001
                        messages.append(ToolMessage(content=f"Tool error: {exc}"))
                else:
                    messages.append(ToolMessage(content=f"Tool not found: {name}"))
            messages.append(AIMessage(content=str(response.get("final_output", ""))))
            return {"messages": messages}

        if engine is not None and hasattr(engine, "complete"):
            text = engine.complete(system=self.prompt, user=task, max_tokens=1024)
            try:
                maybe = json.loads(text)
                if isinstance(maybe, dict) and "tool_calls" in maybe:
                    for call in maybe.get("tool_calls", []):
                        name = str(call.get("name", ""))
                        args = call.get("arguments", {})
                        if name in tool_map:
                            try:
                                output = tool_map[name].invoke(args if isinstance(args, dict) else {})
                                messages.append(ToolMessage(content=str(output)))
                            except Exception as exc:  # noqa: BLE001
                                messages.append(ToolMessage(content=f"Tool error: {exc}"))
                    messages.append(AIMessage(content=str(maybe.get("final_output", ""))))
                    return {"messages": messages}
            except Exception:
                pass
            return {"messages": [AIMessage(content=text)]}

        return {"messages": [AIMessage(content="")]} 


def create_react_agent(model: Any, tools: list[Any], prompt: str) -> _SimpleReactAgent:
    return _SimpleReactAgent(model=model, tools=tools, prompt=prompt)
