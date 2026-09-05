@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start.ps1"
if errorlevel 1 (
  echo.
  echo The workbench could not start. The message above explains what failed.
  pause
)
endlocal
