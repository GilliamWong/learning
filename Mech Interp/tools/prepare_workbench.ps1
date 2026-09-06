$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    throw 'uv was not found. Install uv from https://docs.astral.sh/uv/ and reopen this launcher.'
}

# uv's small Windows launcher can be rejected by Application Control. Use the
# standard CPython venv launcher; this does not change Windows security policy.
$venvRoot = Join-Path $projectRoot '.venv'
$pythonExe = Join-Path $venvRoot 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $basePython = (& $uvCommand.Source python find --system --managed-python 3.12 --cache-dir (Join-Path $projectRoot '.cache\uv')).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Could not locate the installed Python 3.12 runtime.' }
    & $basePython -m venv --without-pip $venvRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the standard Python environment.' }
} elseif (-not (Get-Item -LiteralPath $pythonExe).VersionInfo.FileVersion) {
    $baseHomeLine = Get-Content -LiteralPath (Join-Path $venvRoot 'pyvenv.cfg') | Where-Object { $_ -match '^home\s*=' } | Select-Object -First 1
    $basePython = Join-Path (($baseHomeLine -split '=', 2)[1].Trim()) 'python.exe'
    & $basePython -m venv --upgrade --without-pip $venvRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not repair the Python launcher.' }
}
Write-Host 'Preparing the learning workspace. The first launch downloads its Python packages.'
& $uvCommand.Source sync --frozen --cache-dir (Join-Path $projectRoot '.cache\uv')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
