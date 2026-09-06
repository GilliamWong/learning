@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_vscode.ps1"
if errorlevel 1 (
  echo.
  echo The VS Code workspace could not start. See the message above.
  pause
)
endlocal
