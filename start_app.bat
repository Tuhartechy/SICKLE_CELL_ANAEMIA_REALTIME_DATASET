@echo off
title SickleSense AI - Clinical Triage Portal
echo ============================================================
echo   SICKLESENSE AI: CLINICAL PAIN & VOC TRIAGE SUITE
echo   Backend: FastAPI REST API (Port 8000)
echo   Frontend: React 19 + TypeScript + Vite (Port 5173)
echo ============================================================

set PYTHON=.\sickle-pain\.venv\Scripts\python.exe
if not exist %PYTHON% set PYTHON=python

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000...
start /B "" %PYTHON% -m uvicorn api:app --host 127.0.0.1 --port 8000

timeout /t 2 /nobreak >nul

echo [2/2] Starting SickleSense AI React Frontend on http://localhost:5173...
start http://localhost:5173
cd frontend
npm run dev
