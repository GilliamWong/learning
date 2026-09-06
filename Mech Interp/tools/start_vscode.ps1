param([switch]$NoOpen)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$codeCommand = Get-Command code -ErrorAction SilentlyContinue
if (-not $codeCommand) { throw 'VS Code was not found on PATH. Install VS Code, then reopen this launcher.' }
& (Join-Path $PSScriptRoot 'prepare_workbench.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
& $pythonExe (Join-Path $PSScriptRoot 'package_vscode_extension.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$installed = @(& $codeCommand.Source --list-extensions)
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect VS Code extensions.' }
foreach ($extension in @('ms-python.python', 'ms-toolsai.jupyter')) {
    if ($extension -notin $installed) {
        & $codeCommand.Source --install-extension $extension
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
$sourceHash = Get-Content -LiteralPath (Join-Path $projectRoot '.runtime\vscode-extension-source.sha256') -Raw
$installedMarker = Join-Path $projectRoot '.runtime\vscode-extension-installed.sha256'
$previousHash = if (Test-Path -LiteralPath $installedMarker) { Get-Content -LiteralPath $installedMarker -Raw } else { '' }
if ('local-learning.mech-interp-workbench' -notin $installed -or $previousHash -ne $sourceHash) {
    & $codeCommand.Source --install-extension (Join-Path $projectRoot '.runtime\mech-interp-workbench.vsix') --force
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Set-Content -LiteralPath $installedMarker -Value $sourceHash -NoNewline
}
if (-not $NoOpen) {
    & $codeCommand.Source --new-window $projectRoot
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Host 'VS Code learning workspace ready. Use the Learning dashboard or the Learning button in the status bar.'
