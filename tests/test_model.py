import time
import sys
from llama_cpp import Llama

# Настройки (такие же как у агента)
MODEL_PATH = r"C:\Users\Xandr\Documents\models\qwen2.5-coder-32b-instruct-q3_k_m.gguf"
N_GPU_LAYERS = -1  # -1 означает "загрузить все слои в GPU"
N_CTX = 2048       # Уменьшенный контекст для теста

print(f"--- Запуск теста модели: {MODEL_PATH} ---")

# 1. Загрузка модели
print("Загрузка модели...")
start_load = time.time()
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        verbose=True  # Важно! Покажет, куда грузятся слои (CPU или GPU)
    )
except Exception as e:
    print(f"Ошибка загрузки: {e}")
    sys.exit(1)

end_load = time.time()
print(f"Модель загружена за {end_load - start_load:.2f} сек.")

# 2. Тест генерации
prompt = "Привет, как дела? Напиши короткое стихотворение про код."
print(f"\n--- Отправка промпта: '{prompt}' ---")

start_gen = time.time()

# Потоковая генерация
output_stream = llm.create_chat_completion(
    messages=[{"role": "user", "content": prompt}],
    stream=True,
    max_tokens=100
)

print("\nОтвет модели:")
full_text = ""
for chunk in output_stream:
    if "content" in chunk["choices"][0]["delta"]:
        token = chunk["choices"][0]["delta"]["content"]
        print(token, end="", flush=True)
        full_text += token

end_gen = time.time()
duration = end_gen - start_gen

# 3. Статистика
print("\n\n--- ИТОГИ ТЕСТА ---")
tokens_count = len(full_text.split()) # Приблизительная оценка (слова)
# Для точного подсчета токенов в llama-cpp есть свои методы, но это грубая оценка скорости реакции

print(f"Время генерации: {duration:.2f} сек.")
print(f"Сгенерировано символов: {len(full_text)}")
print(f"Примерная скорость: {len(full_text)/duration:.1f} символов/сек")

# Если скорость низкая (< 10 токенов в сек), проблема в железе или настройках.
# Если скорость высокая (> 30 токенов в сек), проблема была в коде агента.