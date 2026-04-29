# PRE Web GUI — Windows Server Launcher
# PowerShell equivalent of pre-server.sh
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File pre-server.ps1
#        powershell -File pre-server.ps1 --status
#        powershell -File pre-server.ps1 --stop

param(
    [switch]$status,
    [switch]$stop
)

$ErrorActionPreference = "SilentlyContinue"

$WEB_PORT = 7749
$OLLAMA_PORT = if ($env:PRE_PORT) { $env:PRE_PORT } else { 11434 }
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$CONTEXT_FILE = Join-Path $env:USERPROFILE ".pre\context"
$LOG_FILE = Join-Path $env:TEMP "pre-server.log"

# Read context window size
$NUM_CTX = 131072
if (Test-Path $CONTEXT_FILE) {
    $ctx = (Get-Content $CONTEXT_FILE -Raw).Trim()
    if ($ctx -match '^\d+$') { $NUM_CTX = [int]$ctx }
}

# ── Status ──
if ($status) {
    $proc = Get-NetTCPConnection -LocalPort $WEB_PORT -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "PRE server is running on port $WEB_PORT (PID: $($proc.OwningProcess | Select-Object -First 1))"
        Write-Host "Context window: $NUM_CTX tokens"
    } else {
        Write-Host "PRE server is not running"
    }
    exit
}

# ── Stop ──
if ($stop) {
    $conns = Get-NetTCPConnection -LocalPort $WEB_PORT -ErrorAction SilentlyContinue
    if ($conns) {
        $pids = $conns.OwningProcess | Select-Object -Unique
        foreach ($pid in $pids) {
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            Write-Host "Stopped process $pid"
        }
    } else {
        Write-Host "No PRE server process found on port $WEB_PORT"
    }
    exit
}

# ── Start Ollama if not running ──
Write-Host "PRE Server starting..."
Write-Host "  Context window: $NUM_CTX tokens"

$ollamaRunning = $false
try {
    $response = Invoke-RestMethod -Uri "http://127.0.0.1:$OLLAMA_PORT/api/tags" -TimeoutSec 2
    $ollamaRunning = $true
    Write-Host "  Ollama: already running"
} catch {}

if (-not $ollamaRunning) {
    Write-Host "  Starting Ollama..."
    $ollamaExe = Get-Command ollama -ErrorAction SilentlyContinue
    if ($ollamaExe) {
        Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
        # Wait for Ollama to become ready
        $retries = 0
        while ($retries -lt 30) {
            Start-Sleep -Seconds 1
            try {
                Invoke-RestMethod -Uri "http://127.0.0.1:$OLLAMA_PORT/api/tags" -TimeoutSec 2 | Out-Null
                $ollamaRunning = $true
                break
            } catch {}
            $retries++
        }
        if ($ollamaRunning) {
            Write-Host "  Ollama: started"
        } else {
            Write-Host "  WARNING: Ollama did not start within 30 seconds"
        }
    } else {
        Write-Host "  WARNING: ollama not found in PATH. Install from https://ollama.com"
    }
}

# ── Pre-warm the model ──
if ($ollamaRunning) {
    Write-Host "  Pre-warming model..."
    try {
        $body = @{
            model = "pre-gemma4"
            prompt = "hi"
            stream = $false
            options = @{ num_predict = 1; num_ctx = $NUM_CTX }
        } | ConvertTo-Json
        Invoke-RestMethod -Uri "http://127.0.0.1:$OLLAMA_PORT/api/generate" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 120 | Out-Null
        Write-Host "  Model ready"
    } catch {
        Write-Host "  WARNING: model pre-warm failed (will load on first request)"
    }
}

# ── Set Ollama environment variables ──
$env:OLLAMA_KEEP_ALIVE = "24h"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "1"

# ── Start the Node.js web server ──
Write-Host "  Starting web server on port $WEB_PORT..."

Set-Location $SCRIPT_DIR

# Start Node.js and redirect output to log file
$env:PRE_PORT = $OLLAMA_PORT
node server.js 2>&1 | Tee-Object -FilePath $LOG_FILE
