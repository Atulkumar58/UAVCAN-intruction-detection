@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python virtual environment not found at .venv\Scripts\python.exe
  exit /b 1
)
if not exist "processed_data\uavcan_attack_cnn.keras" (
  echo CNN model not found. Train it first with:
  echo .venv\Scripts\python.exe train_uavcan_cnn.py
  exit /b 1
)
if not exist "processed_data\uavcan_attack_model.npz" (
  echo NumPy MLP model not found at processed_data\uavcan_attack_model.npz
  exit /b 1
)
".venv\Scripts\python.exe" "api_uavcan_model.py"
