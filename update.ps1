# update.ps1 - Update PRE (Personal Reasoning Engine) on Windows
#
# Detects whether PRE was installed via git clone or zip download,
# compares local version against the latest release on GitHub,
# and updates accordingly. User data (~/.pre/) is never touched.
#
# Usage:
#   Right-click update.ps1 -> "Run with PowerShell"
#   powershell -ExecutionPolicy Bypass -File update.ps1
#   powershell -File update.ps1 -Yes    # Non-interactive (accept all defaults)

param(
    [switch]$Yes
)

$ErrorActionPreference = "Stop"

# Trap unexpected termination so the window stays open
trap {
    Write-Host ""
    Write-Host "  ERROR: $_" -ForegroundColor Red
    Write-Host "  At: $($_.InvocationInfo.ScriptName):$($_.InvocationInfo.ScriptLineNumber)" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Press Enter to exit..." -ForegroundColor Yellow
    $null = Read-Host
    exit 1
}

# -- Helpers ----------------------------------------------------------------

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

$REPO_URL = "https://github.com/sunkencity999/pre"
$RAW_URL = "https://raw.githubusercontent.com/sunkencity999/pre/main"
$REPO_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$WEB_DIR = Join-Path $REPO_DIR "web"
$ENGINE_DIR = Join-Path $REPO_DIR "engine"
$PRE_PORT = if ($env:PRE_PORT) { $env:PRE_PORT } else { "7749" }

# -- Detect install type ----------------------------------------------------

Step "Checking installation"

$IS_GIT = $false
if (Test-Path (Join-Path $REPO_DIR ".git")) {
    $IS_GIT = $true
    Ok "Install type: git repository"
    # Check for local modifications
    $status = & git -C $REPO_DIR status --porcelain 2>$null
    if ($status) {
        Warn "Local modifications detected."
        & git -C $REPO_DIR status --short
        Write-Host ""
    }
} else {
    Ok "Install type: zip download (no git repository)"
}

# -- Get local version ------------------------------------------------------

$LOCAL_VERSION = "unknown"
$VERSION_FILE = Join-Path $REPO_DIR "VERSION"
if (Test-Path $VERSION_FILE) {
    $LOCAL_VERSION = (Get-Content $VERSION_FILE -Raw).Trim()
}
Ok "Local version: $LOCAL_VERSION"

# -- Get remote version -----------------------------------------------------

Step "Checking for updates"

$REMOTE_VERSION = ""
try {
    $response = Invoke-WebRequest -Uri "$RAW_URL/VERSION" -UseBasicParsing -TimeoutSec 10
    $REMOTE_VERSION = $response.Content.Trim()
} catch {
    # Silently continue
}

if (-not $REMOTE_VERSION) {
    Warn "Could not fetch remote version. Check your internet connection."
    if (-not (Ask-YN "  Continue anyway? [y/N]" "N")) {
        exit 0
    }
    $REMOTE_VERSION = "unknown"
}

Ok "Remote version: $REMOTE_VERSION"

# Compare versions
if (($LOCAL_VERSION -eq $REMOTE_VERSION) -and ($REMOTE_VERSION -ne "unknown")) {
    Ok "You're already on the latest version ($LOCAL_VERSION)."
    if (-not (Ask-YN "  Force update anyway? [y/N]" "N")) {
        Write-Host ""
        Write-Host "  No update needed."
        Write-Host ""
        Write-Host "  Press Enter to exit..." -ForegroundColor Yellow
        $null = Read-Host
        exit 0
    }
} else {
    if ($REMOTE_VERSION -ne "unknown") {
        Ok "Update available: $LOCAL_VERSION -> $REMOTE_VERSION"
    }
}

# -- Check if server is running ---------------------------------------------

$SERVER_WAS_RUNNING = $false
try {
    $null = Invoke-WebRequest -Uri "http://localhost:${PRE_PORT}/api/status" -UseBasicParsing -TimeoutSec 3
    $SERVER_WAS_RUNNING = $true
} catch {
    # Server not running
}

if ($SERVER_WAS_RUNNING) {
    Warn "PRE server is currently running."
    if (Ask-YN "  Stop it for the update? [Y/n]" "Y") {
        Write-Host "  Stopping server..."
        # Try graceful shutdown via the API
        try {
            $null = Invoke-WebRequest -Uri "http://localhost:${PRE_PORT}/api/shutdown" -UseBasicParsing -TimeoutSec 3
        } catch {}
        Start-Sleep -Seconds 1
        # Kill by port if still running
        try {
            $connections = Get-NetTCPConnection -LocalPort $PRE_PORT -State Listen -ErrorAction SilentlyContinue
            if ($connections) {
                $connections | ForEach-Object {
                    Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
                }
            }
        } catch {}
        Start-Sleep -Seconds 1
        Ok "Server stopped."
    } else {
        Warn "Updating while server is running - restart manually after."
    }
}

# -- Perform update ---------------------------------------------------------

Step "Updating PRE"

if ($IS_GIT) {
    # ---- Git-based update ----
    Write-Host "  Pulling latest changes from origin..."

    # Stash local changes if any
    $STASHED = $false
    $status = & git -C $REPO_DIR status --porcelain 2>$null
    if ($status) {
        Warn "Stashing local changes..."
        & git -C $REPO_DIR stash push -m "pre-update-$(Get-Date -Format 'yyyyMMdd-HHmmss')" 2>$null
        $STASHED = $true
    }

    # Pull
    $pullOutput = & git -C $REPO_DIR pull origin main 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  $pullOutput"
        if ($STASHED) {
            Warn "Restoring stashed changes..."
            & git -C $REPO_DIR stash pop 2>$null
        }
        Fail "Git pull failed. Resolve conflicts manually."
    }
    Write-Host "  $pullOutput"

    # Restore stashed changes
    if ($STASHED) {
        Write-Host ""
        Warn "Restoring stashed changes..."
        $popResult = & git -C $REPO_DIR stash pop 2>&1
        if ($LASTEXITCODE -eq 0) {
            Ok "Local changes restored."
        } else {
            Warn "Could not auto-restore local changes. Run 'git stash pop' manually."
        }
    }

    Ok "Git update complete."

} else {
    # ---- Zip-based update ----
    Write-Host "  Downloading latest version from GitHub..."

    $TEMP_DIR = Join-Path $env:TEMP "pre-update-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    New-Item -ItemType Directory -Path $TEMP_DIR -Force | Out-Null
    $ZIP_FILE = Join-Path $TEMP_DIR "pre-latest.zip"
    $EXTRACT_DIR = Join-Path $TEMP_DIR "pre-main"

    # Download
    try {
        Invoke-WebRequest -Uri "$REPO_URL/archive/refs/heads/main.zip" -OutFile $ZIP_FILE -UseBasicParsing
    } catch {
        Remove-Item -Recurse -Force $TEMP_DIR -ErrorAction SilentlyContinue
        Fail "Download failed. Check your internet connection."
    }
    Ok "Downloaded."

    # Extract
    Write-Host "  Extracting..."
    try {
        Expand-Archive -Path $ZIP_FILE -DestinationPath $TEMP_DIR -Force
    } catch {
        Remove-Item -Recurse -Force $TEMP_DIR -ErrorAction SilentlyContinue
        Fail "Extraction failed."
    }

    # The zip extracts to a directory like "pre-main/"
    if (-not (Test-Path $EXTRACT_DIR)) {
        # Try to find the extracted directory
        $found = Get-ChildItem -Path $TEMP_DIR -Directory -Filter "pre-*" | Select-Object -First 1
        if ($found) {
            $EXTRACT_DIR = $found.FullName
        }
    }

    if (-not (Test-Path $EXTRACT_DIR)) {
        Remove-Item -Recurse -Force $TEMP_DIR -ErrorAction SilentlyContinue
        Fail "Could not find extracted files."
    }

    # Copy files, preserving user data
    Write-Host "  Updating files..."

    # Update engine/ (source code, Modelfile, scripts)
    $srcEngine = Join-Path $EXTRACT_DIR "engine"
    if (Test-Path $srcEngine) {
        # Copy engine files, excluding compiled binaries
        Get-ChildItem -Path $srcEngine -Recurse -File | Where-Object {
            $_.Extension -notin @('.o', '.exe') -and $_.Name -ne 'pre' -and $_.Name -ne 'telegram'
        } | ForEach-Object {
            $relativePath = $_.FullName.Substring($srcEngine.Length)
            $destPath = Join-Path $ENGINE_DIR $relativePath
            $destDir = Split-Path -Parent $destPath
            if (-not (Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir -Force | Out-Null
            }
            Copy-Item -Path $_.FullName -Destination $destPath -Force
        }
    }

    # Update web/ (server, tools, frontend)
    $srcWeb = Join-Path $EXTRACT_DIR "web"
    if (Test-Path $srcWeb) {
        # Copy web files, excluding node_modules
        Get-ChildItem -Path $srcWeb -Recurse -File | Where-Object {
            $_.FullName -notmatch [regex]::Escape("node_modules")
        } | ForEach-Object {
            $relativePath = $_.FullName.Substring($srcWeb.Length)
            $destPath = Join-Path $WEB_DIR $relativePath
            $destDir = Split-Path -Parent $destPath
            if (-not (Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir -Force | Out-Null
            }
            Copy-Item -Path $_.FullName -Destination $destPath -Force
        }
    }

    # Update root files (scripts, README, VERSION, etc.)
    $rootFiles = @(
        "VERSION", "README.md", "install.sh", "install.ps1", "install.cmd",
        "Launch PRE.command", "Launch PRE.cmd", "PRE Tray.cmd",
        "Install PRE.command", "Update PRE.command", "Update PRE.cmd",
        "update.sh", "update.ps1", "system.md", "benchmark.sh"
    )
    foreach ($f in $rootFiles) {
        $srcFile = Join-Path $EXTRACT_DIR $f
        if (Test-Path $srcFile) {
            Copy-Item -Path $srcFile -Destination (Join-Path $REPO_DIR $f) -Force
        }
    }

    # Clean up
    Remove-Item -Recurse -Force $TEMP_DIR -ErrorAction SilentlyContinue
    Ok "Files updated."
}

# -- Update dependencies ----------------------------------------------------

Step "Updating dependencies"

$packageJson = Join-Path $WEB_DIR "package.json"
if (Test-Path $packageJson) {
    Write-Host "  Running npm install..."
    Push-Location $WEB_DIR
    try {
        & npm install --silent 2>&1 | Select-Object -Last 3
        Ok "Dependencies updated."
    } finally {
        Pop-Location
    }
} else {
    Warn "web\package.json not found - skipping npm install."
}

# -- Check if Modelfile changed ---------------------------------------------

if ($IS_GIT) {
    try {
        $changed = & git -C $REPO_DIR diff HEAD~1 --name-only 2>$null | Select-String "Modelfile"
        if ($changed) {
            Warn "Modelfile has changed. You may want to recreate the custom model:"
            Write-Host "    ollama create pre-gemma4 -f engine\Modelfile"
        }
    } catch {}
}

# -- Restart server if it was running ----------------------------------------

if ($SERVER_WAS_RUNNING) {
    Step "Restarting server"
    if (Ask-YN "  Restart PRE server? [Y/n]" "Y") {
        Push-Location $WEB_DIR
        try {
            Start-Process -FilePath "node" -ArgumentList "server.js" -WindowStyle Hidden -WorkingDirectory $WEB_DIR
            Start-Sleep -Seconds 2
            try {
                $null = Invoke-WebRequest -Uri "http://localhost:${PRE_PORT}/api/status" -UseBasicParsing -TimeoutSec 3
                Ok "Server restarted on port $PRE_PORT."
            } catch {
                Warn "Server may still be starting. Check http://localhost:${PRE_PORT}"
            }
        } finally {
            Pop-Location
        }
    }
}

# -- Done -------------------------------------------------------------------

Step "Update Complete!"

$NEW_VERSION = "unknown"
if (Test-Path (Join-Path $REPO_DIR "VERSION")) {
    $NEW_VERSION = (Get-Content (Join-Path $REPO_DIR "VERSION") -Raw).Trim()
}

Write-Host ""
Ok "PRE updated to version $NEW_VERSION"
Write-Host ""
Write-Host "  Changes in this update:"
if ($IS_GIT) {
    Write-Host "    git log --oneline -10"
} else {
    Write-Host "    See: $REPO_URL/commits/main"
}
Write-Host ""
Write-Host "  If you encounter issues after updating:"
Write-Host "    1. Restart the server: cd web && node server.js"
Write-Host "    2. Recreate the model: ollama create pre-gemma4 -f engine\Modelfile"
Write-Host "    3. Report issues: $REPO_URL/issues"
Write-Host ""

Write-Host "  Press Enter to exit..." -ForegroundColor Yellow
$null = Read-Host
