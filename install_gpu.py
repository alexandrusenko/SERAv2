import os
import subprocess
import sys
import glob

def find_vcvstem():
    # Ищем vswhere для определения пути к VS
    vswhere_path = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if os.path.exists(vswhere_path):
        result = subprocess.run([vswhere_path, "-latest", "-property", "installationPath"], capture_output=True, text=True)
        vs_path = result.stdout.strip()
        if vs_path:
            return os.path.join(vs_path, "VC", "Auxiliary", "Build", "vcvarsall.bat")
    return None

# 1. Настройки путей (проверь их!)
CUDA_PATH = r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1"
CUDA_ARCH = "90" # Принудительно ставим 90 для RTX 5070 Ti (Blackwell)

# 2. Поиск vcvarsall.bat
vcvars_path = find_vcvstem()
if not vcvars_path:
    print("ОШИБКА: Не удалось найти vcvarsall.bat. Проверь установку Visual Studio.")
    sys.exit(1)

print(f"Найдена среда VS: {vcvars_path}")

# 3. Формируем команду для CMD
# Мы используем cmd /c, чтобы выполнить настройку среды (vcvarsall)
# и затем запустить pip с нужными переменными в той же сессии.
command_str = f"""
call "{vcvars_path}" x64 > nul
set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH}
set CMAKE_GENERATOR=Ninja
set CMAKE_CUDA_COMPILER={CUDA_PATH}/bin/nvcc.exe
set CC=cl.exe
set CXX=cl.exe
{sys.executable} -m pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade --verbose
"""

# 4. Запуск
print("Запуск сборки...")
print(f"Архитектура CUDA: {CUDA_ARCH}")
print(f"Компилятор: {CUDA_PATH}/bin/nvcc.exe")

# Используем shell=True для выполнения составной команды
process = subprocess.Popen(command_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# Вывод логов в реальном времени
while True:
    output = process.stdout.readline()
    if output == '' and process.poll() is not None:
        break
    if output:
        print(output.strip())

return_code = process.poll()
if return_code == 0:
    print("\n=== УСПЕХ! ===")
    print("Проверь работу: python test_speed.py")
else:
    print("\n=== ОШИБКА ===")