@echo off
:: PRE Installer Launcher - double-click this file to install PRE on Windows.
:: This wrapper bypasses PowerShell's execution policy so install.ps1 can run.
:: It also keeps the window open if something goes wrong.

echo.
echo   PRE (Personal Reasoning Engine) - Windows Installer
echo   ====================================================
echo.

:: Find the install.ps1 script in the same directory as this .cmd file
set "SCRIPT_DIR=%~dp0"
set "PS1_FILE=%SCRIPT_DIR%install.ps1"

if not exist "%PS1_FILE%" (
    echo   ERROR: install.ps1 not found in %SCRIPT_DIR%
    echo   Make sure this file is in the same directory as install.ps1
    echo.
    pause
    exit /b 1
)

:: Launch PowerShell with Bypass execution policy
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PS1_FILE%" %*

:: If PowerShell exited with an error, keep the window open
if errorlevel 1 (
    echo.
    echo   Installation encountered an error. See above for details.
    echo.
    pause
)
