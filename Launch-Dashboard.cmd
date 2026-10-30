@echo off
setlocal
cd /d "%~dp0"
title TGNN Fraud Intelligence Platform Launcher

echo ======================================================================
echo    TGNN FRAUD INTELLIGENCE PLATFORM - LAUNCHER
echo ======================================================================
echo.
echo  Which dashboard would you like to run?
echo.
echo  [1] Next-Level Enterprise AML Platform (FastAPI + Web UI) - RECOMMENDED
echo  [2] Classic Streamlit Dashboard (Streamlit App)
echo.

set /p CHOICE="Enter choice (1 or 2, default is 1): "
if "%CHOICE%"=="" set CHOICE=1

if exist ".venv\Scripts\python.exe" (
    set "PY_EXE=.venv\Scripts\python.exe"
) else (
    set "PY_EXE=python"
)

echo.
if "%CHOICE%"=="2" (
    echo Launching Classic Streamlit Dashboard on port 8501...
    "%PY_EXE%" -m streamlit run app/Home.py
) else (
    echo Launching Next-Level Enterprise AML Platform on port 8000...
    start http://127.0.0.1:8000
    "%PY_EXE%" run_dashboard.py
)

echo.
echo Process terminated.
pause
