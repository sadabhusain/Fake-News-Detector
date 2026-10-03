@echo off
setlocal
cd /d "%~dp0"
set "VENV=%~dp0.venv\Scripts\python.exe"
if exist "%VENV%" (
    start "Fake News App" cmd /k "\"%VENV%\" -m streamlit run app.py"
) else (
    start "Fake News App" cmd /k "python -m streamlit run app.py"
)
exit /b
