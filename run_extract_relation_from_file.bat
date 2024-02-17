@echo off
cd /d "%~dp0"


echo Running the project...
python Extract_Relation_From_File.py

echo Press any key to exit...
pause >nul