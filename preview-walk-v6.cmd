@echo off
cd /d "%~dp0"
".venv-preview\Scripts\python.exe" "scripts\preview_hd1910.py" --policy "artifacts\hd1910-walk-v6-symmetry500\hd1910-walk-v6-symmetry500.onnx" --walk-speed 0.2
if errorlevel 1 pause
