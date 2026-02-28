from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path
from queue import Queue
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from sera_agent.main import build_agent

app = FastAPI(title="SERA v2 UI")
LOGGER = logging.getLogger(__name__)

INDEX_HTML = """
<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>SERA v2 UI</title>
    <style>
      body { margin: 0; background: #0b1220; color: #e5e7eb; font-family: Inter, Arial, sans-serif; }
      .container { display: grid; grid-template-columns: 2fr 1fr; gap: 12px; height: 100vh; padding: 12px; box-sizing: border-box; }
      .panel { background: #111827; border: 1px solid #374151; border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; }
      .panel h2 { margin: 0; padding: 12px; font-size: 16px; border-bottom: 1px solid #374151; }
      #status { margin: 12px; padding: 10px 12px; border-radius: 8px; border: 1px solid #4b5563; background: #030712; color: #cbd5e1; font-size: 14px; }
      #status.running { border-color: #2563eb; color: #93c5fd; }
      #status.success { border-color: #16a34a; color: #86efac; }
      #status.error { border-color: #dc2626; color: #fca5a5; }
      #chat, #logs { flex: 1; overflow-y: auto; padding: 12px; white-space: pre-wrap; line-height: 1.4; }
      #chat .user { color: #93c5fd; margin-bottom: 8px; }
      #chat .agent { color: #d1fae5; margin-bottom: 12px; }
      #logs .log { color: #f9a8d4; margin-bottom: 8px; }
      #logs .plan { color: #fde68a; margin-bottom: 8px; }
      #composer { display: flex; gap: 8px; padding: 12px; border-top: 1px solid #374151; }
      #task { flex: 1; background: #030712; color: #f3f4f6; border: 1px solid #4b5563; border-radius: 8px; padding: 8px; }
      button { background: #2563eb; color: white; border: 0; border-radius: 8px; padding: 8px 14px; cursor: pointer; }
      button:disabled { background: #4b5563; cursor: not-allowed; }
    </style>
  </head>
  <body>
    <div class="container">
      <section class="panel">
        <h2>Чат с агентом</h2>
        <div id="status">Статус: ожидание запроса</div>
        <div id="chat"></div>
        <div id="composer">
          <input id="task" placeholder="Введите задачу для SERA v2" />
          <button id="send">Отправить</button>
        </div>
      </section>
      <section class="panel">
        <h2>Детальные логи</h2>
        <div id="logs"></div>
      </section>
    </div>
    <script>
      const chat = document.getElementById("chat");
      const logs = document.getElementById("logs");
      const taskInput = document.getElementById("task");
      const sendButton = document.getElementById("send");
      const status = document.getElementById("status");

      function setStatus(kind, text) {
        status.className = kind;
        status.textContent = "Статус: " + text;
      }

      function addChat(cls, text) {
        const el = document.createElement("div");
        el.className = cls;
        el.textContent = text;
        chat.appendChild(el);
        chat.scrollTop = chat.scrollHeight;
      }

      function addLog(kind, text) {
        const el = document.createElement("div");
        el.className = kind;
        el.textContent = text;
        logs.appendChild(el);
        logs.scrollTop = logs.scrollHeight;
      }

      async function runTask() {
        const task = taskInput.value.trim();
        if (!task) return;

        sendButton.disabled = true;
        logs.innerHTML = "";
        addChat("user", "Вы: " + task);
        setStatus("running", "агент выполняет задачу...");

        const response = await fetch("/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ task })
        });

        if (!response.ok || !response.body) {
          addLog("log", "Не удалось запустить задачу");
          setStatus("error", "не удалось запустить задачу");
          sendButton.disabled = false;
          return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split("\\n");
          buffer = parts.pop() || "";

          for (const line of parts) {
            if (!line.startsWith("data:")) continue;
            const payload = JSON.parse(line.slice(5));
            if (payload.kind === "final") {
              addChat("agent", "SERA: " + payload.message);
            } else if (payload.kind === "status") {
              setStatus(payload.state || "", payload.message);
            } else {
              addLog(payload.kind === "plan" ? "plan" : "log", payload.message);
            }
          }
        }

        sendButton.disabled = false;
      }

      sendButton.addEventListener("click", runTask);
      taskInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") runTask();
      });
    </script>
  </body>
</html>
"""


class RunRequest(BaseModel):
    task: str
    config_path: str = "config.yaml"


@app.on_event("startup")
def initialize_agent() -> None:
    config_path = Path(os.getenv("SERA_UI_CONFIG_PATH", "config.yaml"))
    agent = build_agent(config_path)
    agent.llm.load()
    app.state.agent = agent
    app.state.config_path = config_path
    app.state.run_lock = threading.Lock()
    LOGGER.info("UI agent initialized with config: %s", config_path)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.post("/run")
async def run(request: RunRequest) -> StreamingResponse:
    if not request.task.strip():
        raise HTTPException(status_code=400, detail="Task is empty")
    if Path(request.config_path) != app.state.config_path:
        raise HTTPException(
            status_code=400,
            detail=(
                f"UI service was initialized with '{app.state.config_path}'. "
                "Restart service to use another config_path."
            ),
        )

    queue: Queue[str | None] = Queue()

    def emitter(event: dict[str, str]) -> None:
        queue.put(f"data:{json.dumps(event, ensure_ascii=False)}\n\n")

    def emit_status(state: str, message: str) -> None:
        emitter({"kind": "status", "state": state, "message": message})

    def worker() -> None:
        try:
            emit_status("running", "агент выполняет задачу...")
            with app.state.run_lock:
                app.state.agent.run(request.task, event_handler=emitter)
            emit_status("success", "задача завершена успешно")
        except Exception as exc:  # noqa: BLE001
            emitter({"kind": "log", "message": f"Ошибка выполнения: {exc}"})
            emit_status("error", "задача завершилась ошибкой")
        finally:
            queue.put(None)

    threading.Thread(target=worker, daemon=True).start()

    async def event_stream() -> AsyncIterator[str]:
        while True:
            item = await asyncio.to_thread(queue.get)
            if item is None:
                break
            yield item

    return StreamingResponse(event_stream(), media_type="text/event-stream")
