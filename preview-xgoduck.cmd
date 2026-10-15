@echo off
cd /d "%~dp0"
".venv-preview\Scripts\python.exe" "scripts\preview_xgoduck.py"
if errorlevel 1 pause
