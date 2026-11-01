@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python virtual environment not found at .venv\Scripts\python.exe
  exit /b 1
)
".venv\Scripts\python.exe" "api_uavcan_model.py"
