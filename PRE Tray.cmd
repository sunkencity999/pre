@echo off
:: PRE System Tray - double-click to start the PRE tray icon.
:: This sits in the Windows notification area and lets you
:: start/stop the server, open the browser, and check status.
:: Right-click the tray icon for options.

set "SCRIPT_DIR=%~dp0web"
set "PS1_FILE=%SCRIPT_DIR%\pre-tray.ps1"

if not exist "%PS1_FILE%" (
    echo   ERROR: pre-tray.ps1 not found in %SCRIPT_DIR%
    pause
    exit /b 1
)

:: Launch hidden (no console window) via PowerShell
powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%PS1_FILE%"
