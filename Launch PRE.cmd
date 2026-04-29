@echo off
setlocal enabledelayedexpansion
:: ============================================================
::  PRE - Personal Reasoning Engine
::  Double-click this file to launch PRE on Windows.
:: ============================================================
::
:: This starts Ollama (if needed), launches the web server,
:: and opens your browser to http://localhost:7749.
:: Close this window or press Ctrl+C to stop.

title PRE - Personal Reasoning Engine
cd /d "%~dp0"

set "WEB_DIR=%~dp0web"
set "PRE_PORT=11434"
set "PRE_WEB_PORT=7749"
set "PRE_URL=http://localhost:%PRE_WEB_PORT%"

echo.
echo   PRE - Personal Reasoning Engine
echo   ================================
echo.

:: ---- Check if already running ----
powershell -NoProfile -Command "try { $null = Invoke-RestMethod -Uri '%PRE_URL%/api/status' -TimeoutSec 2; exit 0 } catch { exit 1 }" >nul 2>&1
if !errorlevel!==0 (
    echo   PRE is already running on port %PRE_WEB_PORT%.
    echo   Opening browser...
    start "" "%PRE_URL%"
    echo.
    echo   Press any key to close this window...
    pause >nul
    exit /b 0
)

:: ---- Set Ollama environment ----
set "OLLAMA_KEEP_ALIVE=24h"
set "OLLAMA_NUM_PARALLEL=1"
set "OLLAMA_MAX_LOADED_MODELS=1"

:: NVIDIA GPU optimizations: reduce KV cache VRAM so more layers fit on GPU
where nvidia-smi >nul 2>&1
if !errorlevel!==0 (
    set "OLLAMA_FLASH_ATTENTION=1"
    set "OLLAMA_KV_CACHE_TYPE=q8_0"
    set "OLLAMA_GPU_OVERHEAD=256000000"
    echo   NVIDIA GPU: Flash Attention + q8_0 KV cache enabled
)

:: ---- Start Ollama ----
echo   Checking Ollama...
powershell -NoProfile -Command "try { $null = Invoke-RestMethod -Uri 'http://127.0.0.1:%PRE_PORT%/v1/models' -TimeoutSec 2; exit 0 } catch { exit 1 }" >nul 2>&1
if !errorlevel!==0 goto ollama_ready

:: Ollama not running, try to start it
where ollama >nul 2>&1
if !errorlevel! neq 0 (
    echo   ERROR: Ollama not found. Install from https://ollama.com
    echo.
    pause
    exit /b 1
)

echo   Starting Ollama...
start "" /b ollama serve >nul 2>&1

echo   Waiting for Ollama...
set "RETRIES=0"

:wait_ollama
if !RETRIES! geq 30 (
    echo   WARNING: Ollama may not be ready. Continuing anyway...
    goto ollama_ready
)
timeout /t 1 /nobreak >nul
powershell -NoProfile -Command "try { $null = Invoke-RestMethod -Uri 'http://127.0.0.1:%PRE_PORT%/v1/models' -TimeoutSec 2; exit 0 } catch { exit 1 }" >nul 2>&1
if !errorlevel!==0 (
    echo   Ollama ready.
    goto ollama_ready
)
set /a RETRIES+=1
goto wait_ollama

:ollama_ready

:: ---- Pre-warm model ----
echo   Pre-warming model...
set "CTX=8192"
set "CTX_FILE=%USERPROFILE%\.pre\context"
if exist "%CTX_FILE%" (
    set /p CTX=<"%CTX_FILE%"
)
start "" /b powershell -NoProfile -Command "try { Invoke-RestMethod -Uri 'http://127.0.0.1:%PRE_PORT%/api/generate' -Method Post -Body ('{\"model\":\"pre-gemma4\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1,\"num_ctx\":!CTX!}}') -ContentType 'application/json' -TimeoutSec 120 | Out-Null } catch {}" >nul 2>&1

:: ---- Start web server ----
if not exist "%WEB_DIR%\server.js" (
    echo   ERROR: web\server.js not found. Run the installer first.
    echo.
    pause
    exit /b 1
)

echo   Starting PRE web server on port %PRE_WEB_PORT%...
echo.

:: Open browser after a short delay (suppress all error output)
start "" /b cmd /c "timeout /t 3 /nobreak >nul 2>nul && start "" "%PRE_URL%" >nul 2>nul"

:: Run server in foreground so closing the window stops it
cd /d "%WEB_DIR%"
node server.js

:: If we get here, node exited
echo.
echo   Server stopped.
echo   Press any key to close this window...
pause >nul
