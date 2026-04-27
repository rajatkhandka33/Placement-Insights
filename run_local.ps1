$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$venvPath = Join-Path $PSScriptRoot '.venv'
$pythonExe = Join-Path $venvPath 'Scripts\python.exe'

if (-not (Test-Path $pythonExe)) {
    py -3 -m venv .venv
}

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r requirements.txt
& $pythonExe -m uvicorn app:app --host 127.0.0.1 --port 8000
