Set-Location $PSScriptRoot
uv sync --locked
if ($LASTEXITCODE -eq 0) { uv run --locked distance-annotator }
