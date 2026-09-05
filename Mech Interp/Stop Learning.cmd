@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "tools\launch.py" stop
) else (
  echo The workbench is not installed or running.
)
endlocal
