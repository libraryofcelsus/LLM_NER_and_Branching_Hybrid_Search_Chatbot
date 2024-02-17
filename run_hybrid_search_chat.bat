@echo off
cd /d "%~dp0"
call venv\Scripts\activate

echo Running the project...
python Hybrid_Search_Example.py

echo Press any key to exit...
pause >nul