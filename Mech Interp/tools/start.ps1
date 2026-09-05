param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    throw 'uv was not found. Install uv from https://docs.astral.sh/uv/ and reopen this launcher.'
}
Write-Host 'Preparing the learning workspace. The first launch downloads its Python packages.'
& $uvCommand.Source sync --frozen --cache-dir (Join-Path $projectRoot '.cache\uv')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if ($NoBrowser) {
    & $pythonExe (Join-Path $PSScriptRoot 'launch.py') start --no-browser
} else {
    & $pythonExe (Join-Path $PSScriptRoot 'launch.py') start
}
exit $LASTEXITCODE
