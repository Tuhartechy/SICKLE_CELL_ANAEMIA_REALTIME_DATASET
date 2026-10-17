# PowerShell Launcher for SickleSense AI (FastAPI + React 19 Frontend)
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SICKLESENSE AI: CLINICAL PAIN & VOC TRIAGE SUITE          " -ForegroundColor Green
Write-Host "  Backend: FastAPI REST API (Port 8000)                     " -ForegroundColor Yellow
Write-Host "  Frontend: React 19 + TypeScript + Vite (Port 5173)        " -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan

$VENV_PYTHON = ".\sickle-pain\.venv\Scripts\python.exe"

if (-not (Test-Path $VENV_PYTHON)) {
    $VENV_PYTHON = "python"
}

# Start FastAPI backend in background
Write-Host "[1/2] Starting FastAPI Backend on http://127.0.0.1:8000..." -ForegroundColor Cyan
$backendJob = Start-Process -FilePath $VENV_PYTHON -ArgumentList "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000" -PassThru -WindowStyle Hidden

Start-Sleep -Seconds 2

# Start Vite React frontend
Write-Host "[2/2] Starting SickleSense AI React Frontend on http://localhost:5173..." -ForegroundColor Cyan
Start-Process "http://localhost:5173"

Set-Location frontend
npm run dev
