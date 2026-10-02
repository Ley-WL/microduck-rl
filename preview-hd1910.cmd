@echo off
cd /d "%~dp0"
".venv-preview\Scripts\python.exe" "scripts\preview_hd1910.py"
if errorlevel 1 pause
