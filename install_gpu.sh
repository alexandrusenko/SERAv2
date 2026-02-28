#!/bin/bash

# Настройки путей
CUDA_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1"
CUDA_ARCH="90"

# Путь к vcvarsall.bat (УБЕДИСЬ, ЧТО ПУТЬ ВЕРНЫЙ: Community, Enterprise или Professional)
VCVARS="C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvarsall.bat"

echo "Запуск сборки..."

# Формируем команду. 
# Обрати внимание на тройные кавычки и экранирование -- это критично для пробелов в путях.
# Также меняем set CMAKE_ARGS на set "CMAKE_ARGS=...", чтобы пробелы в путях не ломали строку.

CMD_COMMAND="call \"${VCVARS}\" x64 > nul && set \"CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}\" && set CMAKE_GENERATOR=Ninja && set \"CMAKE_CUDA_COMPILER=${CUDA_PATH}/bin/nvcc.exe\" && set CC=cl.exe && set CXX=cl.exe && python -m pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade --verbose"

cmd.exe /c "${CMD_COMMAND}"

echo "Готово. Если ошибок нет, проверь: python test_speed.py"