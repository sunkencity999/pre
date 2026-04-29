@echo off
:: Update PRE.cmd - Double-click to update PRE on Windows
:: Launches update.ps1 with execution policy bypass so unsigned scripts run.

set "SCRIPT_DIR=%~dp0"
set "PS1_FILE=%SCRIPT_DIR%update.ps1"

if not exist "%PS1_FILE%" (
    echo   ERROR: update.ps1 not found in %SCRIPT_DIR%
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1_FILE%"
