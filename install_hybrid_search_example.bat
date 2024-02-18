@echo off
:: Check if Git is already installed
where git >nul 2>nul
if %errorlevel% equ 0 (
    echo Git is already installed.
) else (
    :: Download Git installer
    echo Downloading Git installer...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/git-for-windows/git/releases/download/v2.41.0.windows.3/Git-2.41.0.3-64-bit.exe' -OutFile '%TEMP%\GitInstaller.exe'"

    :: Install Git
    echo Installing Git...
    %TEMP%\GitInstaller.exe /SILENT /COMPONENTS="icons,ext\reg\shellhere,assoc,assoc_sh"

    :: Delete the installer
    del %TEMP%\GitInstaller.exe
)


echo Cloning the repository...
git clone https://github.com/libraryofcelsus/NER-and-Hybrid-Search-Ai-Chatbot
cd NER-and-Hybrid-Search-Ai-Chatbot

:: Create a virtual environment
python -m venv "venv"

:: Install project dependencies
"venv\Scripts\python" -m pip install -r requirements.txt

echo Press any key to exit...
pause >nul
goto :EOF