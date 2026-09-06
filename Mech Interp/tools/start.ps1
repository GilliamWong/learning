param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'prepare_workbench.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if ($NoBrowser) {
    & $pythonExe (Join-Path $PSScriptRoot 'launch.py') start --no-browser
} else {
    & $pythonExe (Join-Path $PSScriptRoot 'launch.py') start
}
exit $LASTEXITCODE
