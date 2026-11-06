$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error 'uv was not found on PATH.'
}
uv run --locked --no-dev distance-annotator
exit $LASTEXITCODE
