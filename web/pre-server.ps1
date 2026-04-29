# PRE Web GUI - Windows Server Launcher
# PowerShell equivalent of pre-server.sh
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File pre-server.ps1
#        powershell -File pre-server.ps1 --status
#        powershell -File pre-server.ps1 --stop

param(
    [switch]$status,
    [switch]$stop
)

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

# ── Set Ollama environment variables (before starting Ollama) ──
$env:OLLAMA_KEEP_ALIVE = "24h"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "1"

# NVIDIA GPU optimizations: reduce KV cache VRAM so more model layers fit on GPU
$hasNvidia = $false
try { $hasNvidia = [bool](& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null) } catch {}
if ($hasNvidia) {
    # Match KV cache to model quant: <32GB RAM uses q4_K_M model -> q4_0 cache
    $ramGB = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    $kvType = if ($ramGB -lt 32) { "q4_0" } else { "q8_0" }
    $env:OLLAMA_FLASH_ATTENTION = "1"
    $env:OLLAMA_KV_CACHE_TYPE = $kvType
    $env:OLLAMA_GPU_OVERHEAD = "256000000"
}

# ── Start Ollama if not running ──
Write-Host "PRE Server starting..."
Write-Host "  Context window: $NUM_CTX tokens"
if ($hasNvidia) { Write-Host "  NVIDIA GPU: Flash Attention + $($env:OLLAMA_KV_CACHE_TYPE) KV cache enabled" }

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
        # Use /api/chat (not /api/generate) — Gemma 4 is an instruction model and the
        # chat endpoint forces Ollama to allocate the full KV cache at the requested num_ctx.
        $body = '{"model":"pre-gemma4","messages":[{"role":"user","content":"hi"}],"stream":false,"keep_alive":"24h","options":{"num_predict":1,"num_ctx":' + $NUM_CTX + '}}'
        Invoke-RestMethod -Uri "http://127.0.0.1:$OLLAMA_PORT/api/chat" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 300 | Out-Null
        Write-Host "  Model ready"
    } catch {
        Write-Host "  WARNING: model pre-warm failed (will load on first request)"
    }
}

# ── Start the Node.js web server ──
Write-Host "  Starting web server on port $WEB_PORT..."

Set-Location $SCRIPT_DIR

# Verify node_modules exist
if (-not (Test-Path (Join-Path $SCRIPT_DIR "node_modules"))) {
    Write-Host "  ERROR: node_modules not found. Running npm install..." -ForegroundColor Red
    & npm install 2>&1
    if (-not (Test-Path (Join-Path $SCRIPT_DIR "node_modules"))) {
        Write-Host "  FATAL: npm install failed. Cannot start server." -ForegroundColor Red
        Write-Host "  Try running manually: cd $SCRIPT_DIR && npm install" -ForegroundColor Yellow
        Read-Host "  Press Enter to exit"
        exit 1
    }
}

$env:PRE_PORT = $OLLAMA_PORT

# Verify node is available
$nodeExe = Get-Command node -ErrorAction SilentlyContinue
if (-not $nodeExe) {
    Write-Host "  FATAL: node not found in PATH. Install Node.js 18+ from https://nodejs.org" -ForegroundColor Red
    Read-Host "  Press Enter to exit"
    exit 1
}

# Run Node.js in foreground. Use Start-Process -NoNewWindow -Wait so PowerShell
# blocks until Node exits and all stdout/stderr goes to this console window.
$proc = Start-Process -FilePath "node" -ArgumentList "server.js" `
    -WorkingDirectory $SCRIPT_DIR -NoNewWindow -Wait -PassThru

# If we get here, the server exited
if ($proc.ExitCode -ne 0) {
    Write-Host ""
    Write-Host "  Server exited with code $($proc.ExitCode)" -ForegroundColor Red
    Write-Host "  Try running manually: cd $SCRIPT_DIR && node server.js" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "  Press Enter to exit"
}
