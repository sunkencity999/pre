# install.ps1 — Full setup for PRE (Personal Reasoning Engine) on Windows
#
# This script handles everything from Ollama to ready-to-run:
#   0. Checks system requirements (Windows 10/11, NVIDIA GPU, RAM, disk)
#      - 32GB+ RAM: uses q8_0 quantization (~28GB, near-lossless)
#      - 16-31GB RAM: uses q4_K_M quantization (~15GB, good quality)
#   1. Installs/verifies Ollama
#   1b. Configures Ollama environment variables
#   2. Pulls the base model (auto-selected quantization based on RAM)
#   2b. Pulls embedding model (nomic-embed-text for experience ledger + RAG)
#   3. Creates optimized custom model (pre-gemma4) from Modelfile
#   4. Checks/installs Node.js
#   5. Installs Web GUI dependencies (npm install)
#   6. Sets up ~/.pre/ directories (sessions, memory, rag, workflows, triggers)
#   7. Auto-sizes context window based on RAM
#   8. Optional: Whisper + FFmpeg for voice interface
#   9. Optional: auto-start at login
#   10. Pre-warms model into GPU memory
#   11. Creates pre.cmd launcher
#
# Run time: 5-20 minutes depending on internet speed.
# Disk space required: ~28GB for model + negligible for binaries.
#
# Usage (any of these work):
#   Right-click install.ps1 → "Run with PowerShell"
#   powershell -ExecutionPolicy Bypass -File install.ps1
#   powershell -File install.ps1 -Yes    # Non-interactive: accept all defaults

param(
    [switch]$Yes
)

# ── Self-elevate execution policy if needed ───────────────────────────────
# When launched via right-click "Run with PowerShell", the default execution
# policy blocks unsigned scripts. Detect this and re-launch with Bypass.
if ($MyInvocation.Line -notmatch '-ExecutionPolicy') {
    try {
        # Test if we can actually run — if this param block parsed, we're fine.
        # But if the policy would block, PowerShell never gets here. The real
        # fix is the wrapper below for the right-click scenario.
    } catch {}
}

$ErrorActionPreference = "Stop"

# Trap unexpected termination so the window stays open for the user to read
trap {
    Write-Host ""
    Write-Host "  ERROR: $_" -ForegroundColor Red
    Write-Host "  At: $($_.InvocationInfo.ScriptName):$($_.InvocationInfo.ScriptLineNumber)" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Press Enter to exit..." -ForegroundColor Yellow
    $null = Read-Host
    exit 1
}

# ── Helpers ────────────────────────────────────────────────────────────────

function Step($msg) { Write-Host "`n=== $msg ===`n" -ForegroundColor Cyan }
function Ok($msg) { Write-Host "  $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  $msg" -ForegroundColor Yellow }
function Fail($msg) {
    Write-Host "`n  $msg" -ForegroundColor Red
    Write-Host "`n  Press Enter to exit..." -ForegroundColor Yellow
    $null = Read-Host
    exit 1
}

function Ask-YN {
    param([string]$Prompt, [string]$Default = "Y")
    if ($Yes) {
        return ($Default -eq "Y")
    }
    $reply = Read-Host "$Prompt"
    if ($Default -eq "Y") {
        return ($reply -notmatch '^[Nn]')
    }
    return ($reply -match '^[Yy]')
}

$REPO_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$ENGINE_DIR = Join-Path $REPO_DIR "engine"
$WEB_DIR = Join-Path $REPO_DIR "web"
$PRE_DIR = Join-Path $env:USERPROFILE ".pre"
# Model selection happens after RAM detection (Step 0)
$CUSTOM_MODEL = "pre-gemma4"
$PORT = if ($env:PRE_PORT) { $env:PRE_PORT } else { "11434" }

# ============================================================================
# Step 0: System requirements
# ============================================================================
Step "Checking system requirements"

# Windows version
$osInfo = Get-CimInstance Win32_OperatingSystem
$winVer = [System.Environment]::OSVersion.Version
if ($winVer.Major -lt 10) {
    Fail "PRE requires Windows 10 or later. Detected: $($osInfo.Caption)"
}
Ok "OS: $($osInfo.Caption) (Build $($winVer.Build))"

# NVIDIA GPU
$gpuName = ""
try {
    $nvOut = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
    if ($nvOut) {
        $gpuName = $nvOut.Trim()
        Ok "GPU: $gpuName (NVIDIA CUDA)"
    }
} catch {}
if (-not $gpuName) {
    Warn "NVIDIA GPU not detected. PRE requires an NVIDIA GPU with CUDA support."
    Warn "Install the latest NVIDIA drivers from https://www.nvidia.com/drivers"
    if (-not (Ask-YN "  Continue without GPU verification? [y/N]" "N")) { exit 1 }
}

# RAM + model selection
$ramBytes = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
$ramGB = [math]::Floor($ramBytes / 1GB)

if ($ramGB -lt 16) {
    Fail "PRE requires at least 16GB RAM. Detected: ${ramGB}GB"
}

# Select quantization based on available RAM
# q8_0 (~28GB weights) needs 32GB+ to leave room for KV cache + OS
# q4_K_M (~15GB weights) fits comfortably in 16-31GB
if ($ramGB -ge 32) {
    $BASE_MODEL = "gemma4:26b-a4b-it-q8_0"
    $QUANT = "q8_0"
    $MODEL_SIZE_GB = 28
    $MODEL_SIZE_LABEL = "~28GB"
    Ok "RAM: ${ramGB}GB — using q8_0 quantization (near-lossless, ${MODEL_SIZE_LABEL})"
} else {
    $BASE_MODEL = "gemma4:26b-a4b-it-q4_K_M"
    $QUANT = "q4_K_M"
    $MODEL_SIZE_GB = 15
    $MODEL_SIZE_LABEL = "~15GB"
    Warn "RAM: ${ramGB}GB — using q4_K_M quantization (${MODEL_SIZE_LABEL}, good quality)"
    Warn "Upgrade to 32GB+ RAM to use the higher-quality q8_0 quantization."
}

if ($ramGB -lt 32) {
    # Already warned above
} elseif ($ramGB -lt 64) {
    Warn "RAM: ${ramGB}GB - functional, but 64GB+ recommended for large context windows."
} else {
    Ok "RAM: ${ramGB}GB"
}

# Auto-size context window based on RAM
# q4_K_M uses ~15GB for weights, q8_0 uses ~28GB. KV cache scales with context.
if ($ramGB -ge 96) {
    $OPTIMAL_CTX = 131072   # 128K
} elseif ($ramGB -ge 64) {
    $OPTIMAL_CTX = 65536    # 64K
} elseif ($ramGB -ge 48) {
    $OPTIMAL_CTX = 32768    # 32K
} elseif ($ramGB -ge 36) {
    $OPTIMAL_CTX = 16384    # 16K
} elseif ($ramGB -ge 24) {
    $OPTIMAL_CTX = 16384    # 16K (q4_K_M leaves more headroom)
} elseif ($ramGB -ge 20) {
    $OPTIMAL_CTX = 8192     # 8K
} else {
    $OPTIMAL_CTX = 4096     # 4K (tight fit with q4_K_M on 16GB)
}
$CTX_HUMAN = "$([math]::Floor($OPTIMAL_CTX / 1024))K"
Ok "Context window: $CTX_HUMAN ($OPTIMAL_CTX tokens, auto-sized for ${ramGB}GB RAM)"

# Disk space
$drive = (Get-Item $REPO_DIR).PSDrive
$availGB = [math]::Floor((Get-PSDrive $drive.Name).Free / 1GB)
$diskNeeded = $MODEL_SIZE_GB + 5
if ($availGB -lt $diskNeeded) {
    Warn "Disk: ${availGB}GB available - need ~${MODEL_SIZE_GB}GB for $QUANT model download."
    if (-not (Ask-YN "  Continue anyway? [y/N]" "N")) { exit 1 }
} else {
    Ok "Disk: ${availGB}GB available"
}

Write-Host ""
Ok "System requirements met."

# ============================================================================
# Step 1: Ollama
# ============================================================================
Step "Checking Ollama"

$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollamaCmd) {
    $ollamaVer = & ollama --version 2>&1 | Select-Object -First 1
    Ok "Ollama installed: $ollamaVer"
} else {
    Write-Host "  Ollama is not installed."
    $wingetAvail = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetAvail) {
        if (Ask-YN "  Install Ollama via winget? [Y/n]" "Y") {
            Write-Host "  Installing Ollama..."
            & winget install Ollama.Ollama --accept-package-agreements --accept-source-agreements
            # Refresh PATH
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
            $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
            if (-not $ollamaCmd) {
                Fail "Ollama installation failed. Please install manually from https://ollama.com/download"
            }
            Ok "Ollama installed."
        } else {
            Write-Host ""
            Write-Host "  Install Ollama:"
            Write-Host "    winget install Ollama.Ollama"
            Write-Host "    - or -"
            Write-Host "    Download from https://ollama.com/download"
            exit 1
        }
    } else {
        Write-Host ""
        Write-Host "  Install Ollama from https://ollama.com/download"
        exit 1
    }
}

# Ensure Ollama is running
$ollamaRunning = $false
try {
    $null = Invoke-RestMethod -Uri "http://127.0.0.1:${PORT}/api/tags" -TimeoutSec 3
    $ollamaRunning = $true
} catch {}

if (-not $ollamaRunning) {
    Write-Host "  Starting Ollama..."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    $retries = 0
    while ($retries -lt 30) {
        Start-Sleep -Seconds 1
        try {
            $null = Invoke-RestMethod -Uri "http://127.0.0.1:${PORT}/api/tags" -TimeoutSec 2
            $ollamaRunning = $true
            break
        } catch {}
        $retries++
    }
    if (-not $ollamaRunning) {
        Fail "Could not start Ollama. Run 'ollama serve' in another terminal, then re-run this script."
    }
}
Ok "Ollama running on port $PORT."

# ============================================================================
# Step 1b: Ollama environment variables
# ============================================================================
Step "Configuring Ollama environment"

$envVars = @{
    "OLLAMA_KEEP_ALIVE" = "24h"
    "OLLAMA_NUM_PARALLEL" = "1"
    "OLLAMA_MAX_LOADED_MODELS" = "1"
}

foreach ($key in $envVars.Keys) {
    $current = [Environment]::GetEnvironmentVariable($key, "User")
    if ($current -ne $envVars[$key]) {
        [Environment]::SetEnvironmentVariable($key, $envVars[$key], "User")
        [Environment]::SetEnvironmentVariable($key, $envVars[$key], "Process")
    }
}
Ok "Ollama environment set (KEEP_ALIVE=24h, NUM_PARALLEL=1, MAX_LOADED=1)"

# ============================================================================
# Step 2: Pull base model
# ============================================================================
Step "Pulling base model ($BASE_MODEL)"

$modelList = & ollama list 2>&1
$modelTag = $BASE_MODEL -replace '.*:', ''
if ($modelList -match "gemma4.*$([regex]::Escape($modelTag))") {
    Ok "Base model already available."
} else {
    Write-Host "  Downloading $MODEL_SIZE_LABEL model ($QUANT). This may take a while..."
    Write-Host "  (Download is resumable - safe to interrupt and re-run)"
    & ollama pull $BASE_MODEL
    if ($LASTEXITCODE -ne 0) { Fail "Failed to pull $BASE_MODEL" }
    Ok "Base model downloaded."
}

# ============================================================================
# Step 2b: Pull embedding model
# ============================================================================
Step "Pulling embedding model (nomic-embed-text)"

$modelList = & ollama list 2>&1
if ($modelList -match "nomic-embed-text") {
    Ok "Embedding model already available."
} else {
    Write-Host "  Downloading nomic-embed-text (~274MB)..."
    & ollama pull nomic-embed-text
    if ($LASTEXITCODE -ne 0) {
        Warn "Failed to pull nomic-embed-text - experience ledger and RAG will use keyword search fallback."
    } else {
        Ok "Embedding model downloaded."
    }
}

# ============================================================================
# Step 3: Create custom model from Modelfile
# ============================================================================
Step "Creating optimized model ($CUSTOM_MODEL)"

# The Modelfile uses q8_0 as its FROM line. If we're using q4_K_M, generate
# a modified Modelfile on-the-fly with the correct base model.
$modelfile = Join-Path $ENGINE_DIR "Modelfile"
if (-not (Test-Path $modelfile)) {
    Fail "Modelfile not found at $modelfile"
}

$effectiveModelfile = $modelfile
if ($QUANT -ne "q8_0") {
    $tempModelfile = Join-Path $env:TEMP "pre-Modelfile-$QUANT"
    $content = Get-Content $modelfile -Raw
    $content = $content -replace 'FROM gemma4:26b-a4b-it-q8_0', "FROM $BASE_MODEL"
    Set-Content -Path $tempModelfile -Value $content -NoNewline
    $effectiveModelfile = $tempModelfile
    Ok "Using modified Modelfile with $QUANT base"
}

$modelList = & ollama list 2>&1
if ($modelList -match $CUSTOM_MODEL) {
    Ok "Custom model already exists."
    if (Ask-YN "  Recreate from Modelfile? [y/N]" "N") {
        & ollama create $CUSTOM_MODEL -f $effectiveModelfile
        if ($LASTEXITCODE -ne 0) { Fail "Failed to create $CUSTOM_MODEL" }
        Ok "Custom model recreated."
    }
} else {
    & ollama create $CUSTOM_MODEL -f $effectiveModelfile
    if ($LASTEXITCODE -ne 0) { Fail "Failed to create $CUSTOM_MODEL" }
    Ok "Custom model created ($QUANT, dynamic context, optimized batch size)."
}

# ============================================================================
# Step 4: Node.js
# ============================================================================
Step "Checking Node.js"

$nodeCmd = Get-Command node -ErrorAction SilentlyContinue
if ($nodeCmd) {
    $nodeVer = & node --version
    $nodeMajor = [int]($nodeVer -replace 'v' -replace '\..*')
    if ($nodeMajor -ge 18) {
        Ok "Node.js: $nodeVer"
    } else {
        Warn "Node.js $nodeVer is too old - need v18+."
        $wingetAvail = Get-Command winget -ErrorAction SilentlyContinue
        if ($wingetAvail) {
            if (Ask-YN "  Install Node.js LTS via winget? [Y/n]" "Y") {
                & winget install OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements
                $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
            }
        } else {
            Write-Host "  Install Node.js from https://nodejs.org/"
            exit 1
        }
    }
} else {
    Write-Host "  Node.js not found."
    $wingetAvail = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetAvail) {
        if (Ask-YN "  Install Node.js LTS via winget? [Y/n]" "Y") {
            & winget install OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        } else {
            Write-Host "  Install Node.js from https://nodejs.org/"
            exit 1
        }
    } else {
        Write-Host "  Install Node.js from https://nodejs.org/"
        exit 1
    }
}

# ============================================================================
# Step 5: Web GUI dependencies
# ============================================================================
Step "Setting up Web GUI"

if (Test-Path (Join-Path $WEB_DIR "package.json")) {
    Write-Host "  Installing web dependencies..."
    Push-Location $WEB_DIR
    & npm install --silent 2>&1 | Select-Object -Last 3
    Pop-Location
    Ok "Web GUI dependencies installed"
} else {
    Warn "Web GUI not found at $WEB_DIR - skipping."
}

# ============================================================================
# Step 6: Set up ~/.pre/ directories
# ============================================================================
Step "Setting up PRE data directories"

$dirs = @(
    "sessions", "memory", "memory\experience", "checkpoints",
    "artifacts", "rag", "workflows", "custom_tools"
)

foreach ($d in $dirs) {
    $fullPath = Join-Path $PRE_DIR $d
    if (-not (Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
    }
    Ok "Created $fullPath"
}

# Write context window config
$contextFile = Join-Path $PRE_DIR "context"
Set-Content -Path $contextFile -Value $OPTIMAL_CTX -NoNewline
Ok "Context window: $CTX_HUMAN -> ~/.pre/context"

# Create default config files if they don't exist
$defaultConfigs = @{
    "hooks.json"    = '{"hooks":[]}'
    "mcp.json"      = '{"servers":{}}'
    "cron.json"     = '{"jobs":[]}'
    "triggers.json" = '{"triggers":[]}'
}

foreach ($file in $defaultConfigs.Keys) {
    $filePath = Join-Path $PRE_DIR $file
    if (-not (Test-Path $filePath)) {
        Set-Content -Path $filePath -Value $defaultConfigs[$file]
        Ok "Created $filePath"
    }
}

# ============================================================================
# Step 7: Optional — Whisper + FFmpeg for voice interface
# ============================================================================
Step "Voice Interface Setup (optional)"

$hasWhisper = Get-Command whisper -ErrorAction SilentlyContinue
$hasFfmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue

if ($hasWhisper -and $hasFfmpeg) {
    Ok "Whisper and FFmpeg already installed."
} else {
    Write-Host ""
    Write-Host "  Voice interface requires Whisper (STT) and FFmpeg (audio conversion)."
    if (-not $hasWhisper) { Write-Host "  - Whisper: not installed (pip install openai-whisper)" }
    if (-not $hasFfmpeg) { Write-Host "  - FFmpeg: not installed" }
    Write-Host ""

    $wingetAvail = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetAvail -and -not $hasFfmpeg) {
        if (Ask-YN "  Install FFmpeg via winget? [y/N]" "N") {
            & winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
            Ok "FFmpeg installed."
        }
    }

    if (-not $hasWhisper) {
        Write-Host "  To install Whisper later: pip install openai-whisper"
    }
}

# ============================================================================
# Step 8: Optional — Auto-start at login
# ============================================================================
Step "Auto-start Setup (optional)"

$startupDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
$vbsPath = Join-Path $startupDir "PRE-Server.vbs"
$ps1Path = Join-Path $WEB_DIR "pre-server.ps1"

if (Test-Path $vbsPath) {
    Ok "Auto-start already configured."
} else {
    Write-Host "  PRE can start automatically when you log in."
    Write-Host "  (Creates a VBScript in your Startup folder that launches the server hidden)"
    Write-Host ""
    if (Ask-YN "  Enable auto-start? [y/N]" "N") {
        $webDirWin = $WEB_DIR -replace '/', '\'
        $ps1PathWin = $ps1Path -replace '/', '\'
        $vbs = "Set WshShell = CreateObject(`"WScript.Shell`")`r`n"
        $vbs += "WshShell.CurrentDirectory = `"$webDirWin`"`r`n"
        $vbs += "WshShell.Run `"powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"`"$ps1PathWin`"`"`", 0, False`r`n"
        Set-Content -Path $vbsPath -Value $vbs
        Ok "Auto-start enabled: $vbsPath"
    } else {
        Write-Host "  Skipped. Enable later via the PRE settings panel."
    }
}

# ============================================================================
# Step 9: Pre-warm model
# ============================================================================
Step "Pre-warming model"

Write-Host "  Loading model into GPU memory (this may take 30-60 seconds)..."
try {
    $body = @{
        model = "pre-gemma4"
        prompt = "hi"
        stream = $false
        options = @{ num_predict = 1; num_ctx = $OPTIMAL_CTX }
    } | ConvertTo-Json
    $null = Invoke-RestMethod -Uri "http://127.0.0.1:${PORT}/api/generate" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 120
    Ok "Model loaded and ready."
} catch {
    Warn "Pre-warm failed (model will load on first request): $($_.Exception.Message)"
}

# ============================================================================
# Step 10: Create pre.cmd launcher
# ============================================================================
Step "Creating PRE launcher"

$preCmdPath = Join-Path $env:USERPROFILE ".local\bin\pre.cmd"
$preCmdDir = Split-Path $preCmdPath
if (-not (Test-Path $preCmdDir)) { New-Item -ItemType Directory -Path $preCmdDir -Force | Out-Null }

$preCmdContent = @"
@echo off
:: PRE launcher — starts Ollama + web server, opens browser
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$($ps1Path -replace '/', '\')"
"@
Set-Content -Path $preCmdPath -Value $preCmdContent
Ok "Created $preCmdPath"

# Add to PATH if needed
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notmatch [regex]::Escape($preCmdDir)) {
    [Environment]::SetEnvironmentVariable("Path", "$preCmdDir;$userPath", "User")
    $env:Path = "$preCmdDir;$env:Path"
    Ok "Added $preCmdDir to PATH"
    Warn "Open a new terminal for the PATH change to take effect."
}

# ============================================================================
# Done!
# ============================================================================
Step "Installation Complete!"

Write-Host ""
Ok "PRE is ready to use!"
Write-Host ""
Write-Host "  Quick start:" -ForegroundColor Cyan
Write-Host "    cd $WEB_DIR"
Write-Host "    node server.js"
Write-Host "    Open http://localhost:7749 in your browser"
Write-Host ""
Write-Host "  Or use the launcher:" -ForegroundColor Cyan
Write-Host "    pre"
Write-Host ""
Write-Host "  Configuration:" -ForegroundColor Cyan
Write-Host "    Context window: $CTX_HUMAN ($OPTIMAL_CTX tokens)"
Write-Host "    Data directory: $PRE_DIR"
Write-Host "    Model: $CUSTOM_MODEL (Gemma 4 26B $QUANT)"
if ($gpuName) {
    Write-Host "    GPU: $gpuName"
}
Write-Host ""
Write-Host "  Note: The CLI engine (pre.m) is macOS-only." -ForegroundColor Yellow
Write-Host "  The Web GUI provides the full PRE experience on Windows." -ForegroundColor Yellow
Write-Host ""
Write-Host "  Press Enter to exit..." -ForegroundColor Gray
$null = Read-Host
