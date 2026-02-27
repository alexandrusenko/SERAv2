# SERA v2 — Self-Improving Local Agent (Python 3.14)

## Что это

Промышленный каркас локального AI-агента с упором на **качество, надежность, устойчивость к тупикам и забывчивости**:

- многошаговое планирование + исполнение;
- долговременная память (SQLite);
- безопасный реестр инструментов;
- self-improvement: агент может сгенерировать и подключить недостающий tool «на лету»;
- подробные структурированные логи.

## Почему `llama-cpp-python`

Для Windows + CUDA + GGUF (Qwen 2.5 / 3.5) самый практичный стек — `llama-cpp-python` с CUDA-сборкой.
Он работает с вашими `*.gguf` моделями и поддерживает offload на GPU (`n_gpu_layers=-1`).

## Рекомендуемая модель из ваших

1. **Qwen3.5-27B-Q4_K_M.gguf** — основной brain (лучший баланс reasoning/качество).
2. **qwen2.5-coder-32b-instruct-q4_k_m.gguf** — для задач кодогенерации/рефакторинга.
3. **qwen2.5-14b-instruct-q5_k_m.gguf** — fallback/быстрые подзадачи.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
python -m sera_agent.main --config config.yaml "Собери план миграции проекта на Python 3.14"
```

## UI в реальном времени

Теперь с агентом можно общаться через веб-интерфейс: чат слева, детальные логи выполнения справа.

```bash
uvicorn sera_agent.ui.server:app --host 0.0.0.0 --port 8000
```

После запуска откройте `http://localhost:8000`.


## Типичные проблемы при запуске

- `Failed to load model from file` / `Failed to initialize llama-cpp model`:
  - проверьте, что `runtime.model_path` указывает на существующий `.gguf` файл;
  - убедитесь, что файл не повреждён и совместим с вашей версией `llama-cpp-python`;
  - для больших моделей проверьте доступную RAM/VRAM и параметры offload.
  - если видите `AttributeError: 'LlamaModel' object has no attribute 'sampler'`, обычно это проблема версии `llama-cpp-python`/Python; обновите `llama-cpp-python` и при необходимости используйте Python 3.12/3.13.

## Безопасность

По умолчанию shell-исполнение отключено (`allow_shell: false`). Включайте только если осознанно доверяете окружению.

## Архитектура

Подробно: `docs/ARCHITECTURE_RU.md`.
