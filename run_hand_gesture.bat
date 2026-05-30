@echo off
setlocal

set APP_DIR=%~dp0
set PYTHON=%APP_DIR%.venv\Scripts\python.exe

if not exist "%PYTHON%" (
  echo Virtual environment not found.
  echo Run: python -m venv .venv
  echo Then: .venv\Scripts\python.exe -m pip install -r requirements.txt
  exit /b 1
)

"%PYTHON%" "%APP_DIR%hand_gesture_app.py" %*
