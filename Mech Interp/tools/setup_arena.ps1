$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$uvCommand = Get-Command uv -ErrorAction Stop
& (Join-Path $projectRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'fetch_arena_dependencies.py')
if ($LASTEXITCODE -ne 0) { throw 'ARENA source files could not be prepared. Run Start Learning.cmd first if the opening environment is missing.' }
$arenaEnv = Join-Path $projectRoot 'arena\.venv'
$arenaPython = Join-Path $arenaEnv 'Scripts\python.exe'
$basePython = (& $uvCommand.Source python find --system --managed-python 3.12 --cache-dir (Join-Path $projectRoot '.cache\uv')).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 could not be located.' }
if (-not (Test-Path -LiteralPath $arenaPython)) {
    & $basePython -m venv --without-pip $arenaEnv
} elseif (-not (Get-Item -LiteralPath $arenaPython).VersionInfo.FileVersion) {
    & $basePython -m venv --upgrade --without-pip $arenaEnv
}
if ($LASTEXITCODE -ne 0) { throw 'The standard Python environment could not be prepared.' }
& $uvCommand.Source sync --project arena --frozen --cache-dir (Join-Path $projectRoot '.cache\uv')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$circuitsEnv = Join-Path $projectRoot 'arena\circuits\.venv'
$circuitsPython = Join-Path $circuitsEnv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $circuitsPython)) {
    & $basePython -m venv --without-pip $circuitsEnv
} elseif (-not (Get-Item -LiteralPath $circuitsPython).VersionInfo.FileVersion) {
    & $basePython -m venv --upgrade --without-pip $circuitsEnv
}
if ($LASTEXITCODE -ne 0) { throw 'The SAE circuits environment could not be prepared.' }
& $uvCommand.Source sync --project arena/circuits --frozen --cache-dir (Join-Path $projectRoot '.cache\uv')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $projectRoot '.venv\Scripts\python.exe') (Join-Path $PSScriptRoot 'register_arena_kernel.py')
exit $LASTEXITCODE
