@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m streamlit run app/Home.py
) else (
  python -m streamlit run app/Home.py
)
if errorlevel 1 pause
