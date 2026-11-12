$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonCandidates = @(
    (Get-Command python -ErrorAction SilentlyContinue).Source,
    (Get-Command py -ErrorAction SilentlyContinue).Source,
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
) | Where-Object { $_ -and (Test-Path $_) }

if (-not $pythonCandidates) {
    throw 'Python 3.11+ was not found. Install Python from https://www.python.org/downloads/ and rerun this script.'
}

$python = $pythonCandidates[0]
Set-Location $projectRoot
if (-not (Test-Path .venv\Scripts\python.exe)) {
    & $python -m venv .venv
}
& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\python.exe -m pip install -r requirements.txt
& .venv\Scripts\python.exe -m spacy download en_core_web_sm
Write-Host 'Environment ready. Activate with .\.venv\Scripts\Activate.ps1'

