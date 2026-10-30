@echo off
cd /d "%~dp0"
echo ======================================================================
echo   Launching Next-Level TGNN Fraud Intelligence Platform...
echo ======================================================================
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" run_dashboard.py %*
) else (
  python run_dashboard.py %*
)
if errorlevel 1 pause
