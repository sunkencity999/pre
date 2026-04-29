# PRE System Tray - Windows notification area app for managing the PRE server
# Provides start/stop/restart, status indicator, and quick browser launch.
# Mirrors the macOS menu bar app (engine/pre-menubar.swift).
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File pre-tray.ps1
#   Double-click "PRE Tray.cmd" in the repo root
#
# Requirements: Windows 10+, .NET Framework (built into Windows)

param()

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# -- Configuration ----------------------------------------------------------

$WEB_PORT = if ($env:PRE_WEB_PORT) { $env:PRE_WEB_PORT } else { "7749" }
$OLLAMA_PORT = if ($env:PRE_PORT) { $env:PRE_PORT } else { "11434" }
$PRE_URL = "http://localhost:$WEB_PORT"
$OLLAMA_URL = "http://127.0.0.1:$OLLAMA_PORT"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO_DIR = Split-Path -Parent $SCRIPT_DIR
$WEB_DIR = $SCRIPT_DIR
$CHECK_INTERVAL_MS = 5000

# -- State ------------------------------------------------------------------

$script:serverRunning = $false
$script:ollamaRunning = $false
$script:serverProcess = $null

# -- HTTP check helper ------------------------------------------------------

function Test-Endpoint {
    param([string]$Url)
    try {
        $req = [System.Net.WebRequest]::Create($Url)
        $req.Timeout = 3000
        $req.Method = "GET"
        $resp = $req.GetResponse()
        $code = [int]$resp.StatusCode
        $resp.Close()
        return ($code -ge 200 -and $code -lt 400)
    } catch {
        return $false
    }
}

# -- Create tray icon -------------------------------------------------------

# Create a simple PRE icon (green/red circle indicator)
function New-TrayIcon {
    param([bool]$Running)
    $bmp = New-Object System.Drawing.Bitmap(16, 16)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias

    # Background - transparent
    $g.Clear([System.Drawing.Color]::Transparent)

    # Draw "P" letter
    $font = New-Object System.Drawing.Font("Segoe UI", 8, [System.Drawing.FontStyle]::Bold)
    $brush = [System.Drawing.Brushes]::White
    $g.FillRectangle([System.Drawing.Brushes]::DarkSlateBlue, 0, 0, 16, 16)
    $g.DrawString("P", $font, $brush, 2, 0)

    # Status dot (bottom-right corner)
    if ($Running) {
        $g.FillEllipse([System.Drawing.Brushes]::Lime, 9, 9, 6, 6)
    } else {
        $g.FillEllipse([System.Drawing.Brushes]::Red, 9, 9, 6, 6)
    }

    $g.Dispose()
    $font.Dispose()

    $hIcon = $bmp.GetHicon()
    $icon = [System.Drawing.Icon]::FromHandle($hIcon)
    return $icon
}

# -- Build context menu ------------------------------------------------------

$notifyIcon = New-Object System.Windows.Forms.NotifyIcon
$notifyIcon.Text = "PRE - Personal Reasoning Engine"
$notifyIcon.Icon = New-TrayIcon -Running $false
$notifyIcon.Visible = $true

$contextMenu = New-Object System.Windows.Forms.ContextMenuStrip

# Status items (disabled, informational)
$statusItem = New-Object System.Windows.Forms.ToolStripMenuItem
$statusItem.Text = "Server: checking..."
$statusItem.Enabled = $false
$contextMenu.Items.Add($statusItem) | Out-Null

$ollamaItem = New-Object System.Windows.Forms.ToolStripMenuItem
$ollamaItem.Text = "Ollama: checking..."
$ollamaItem.Enabled = $false
$contextMenu.Items.Add($ollamaItem) | Out-Null

$contextMenu.Items.Add((New-Object System.Windows.Forms.ToolStripSeparator)) | Out-Null

# Open PRE in browser
$openItem = New-Object System.Windows.Forms.ToolStripMenuItem
$openItem.Text = "Open PRE"
$openItem.Font = New-Object System.Drawing.Font($openItem.Font, [System.Drawing.FontStyle]::Bold)
$openItem.Add_Click({
    if ($script:serverRunning) {
        Start-Process $PRE_URL
    } else {
        Start-Server
        Start-Sleep -Seconds 3
        Start-Process $PRE_URL
    }
})
$contextMenu.Items.Add($openItem) | Out-Null

$contextMenu.Items.Add((New-Object System.Windows.Forms.ToolStripSeparator)) | Out-Null

# Start Server
$startItem = New-Object System.Windows.Forms.ToolStripMenuItem
$startItem.Text = "Start Server"
$startItem.Add_Click({ Start-Server })
$contextMenu.Items.Add($startItem) | Out-Null

# Stop Server
$stopItem = New-Object System.Windows.Forms.ToolStripMenuItem
$stopItem.Text = "Stop Server"
$stopItem.Add_Click({ Stop-Server })
$contextMenu.Items.Add($stopItem) | Out-Null

# Restart Server
$restartItem = New-Object System.Windows.Forms.ToolStripMenuItem
$restartItem.Text = "Restart Server"
$restartItem.Add_Click({
    Stop-Server
    Start-Sleep -Seconds 2
    Start-Server
})
$contextMenu.Items.Add($restartItem) | Out-Null

$contextMenu.Items.Add((New-Object System.Windows.Forms.ToolStripSeparator)) | Out-Null

# Open terminal
$termItem = New-Object System.Windows.Forms.ToolStripMenuItem
$termItem.Text = "Open Terminal Here"
$termItem.Add_Click({
    Start-Process "cmd.exe" -ArgumentList "/k cd /d `"$WEB_DIR`"" -WorkingDirectory $WEB_DIR
})
$contextMenu.Items.Add($termItem) | Out-Null

$contextMenu.Items.Add((New-Object System.Windows.Forms.ToolStripSeparator)) | Out-Null

# Quit
$quitItem = New-Object System.Windows.Forms.ToolStripMenuItem
$quitItem.Text = "Quit PRE Tray"
$quitItem.Add_Click({
    $script:timer.Stop()
    $notifyIcon.Visible = $false
    $notifyIcon.Dispose()
    [System.Windows.Forms.Application]::Exit()
})
$contextMenu.Items.Add($quitItem) | Out-Null

$notifyIcon.ContextMenuStrip = $contextMenu

# Double-click opens browser
$notifyIcon.Add_DoubleClick({
    if ($script:serverRunning) {
        Start-Process $PRE_URL
    } else {
        Start-Server
        Start-Sleep -Seconds 3
        Start-Process $PRE_URL
    }
})

# -- Server control functions ------------------------------------------------

function Start-Server {
    $statusItem.Text = "Server: starting..."

    # Ensure Ollama is running
    if (-not $script:ollamaRunning) {
        $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
        if ($ollamaCmd) {
            Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
            # Wait for Ollama
            for ($i = 0; $i -lt 15; $i++) {
                Start-Sleep -Seconds 1
                if (Test-Endpoint "$OLLAMA_URL/v1/models") { break }
            }
        }
    }

    # Start the Node.js server
    $serverJS = Join-Path $WEB_DIR "server.js"
    if (Test-Path $serverJS) {
        $script:serverProcess = Start-Process node -ArgumentList "server.js" -WorkingDirectory $WEB_DIR -WindowStyle Hidden -PassThru
    }

    Start-Sleep -Seconds 2
    Update-Status
}

function Stop-Server {
    $statusItem.Text = "Server: stopping..."

    # Kill any node process on our port
    try {
        $connections = Get-NetTCPConnection -LocalPort $WEB_PORT -State Listen -ErrorAction SilentlyContinue
        foreach ($conn in $connections) {
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    } catch {}

    # Also kill our tracked process
    if ($script:serverProcess -and -not $script:serverProcess.HasExited) {
        try { $script:serverProcess.Kill() } catch {}
    }
    $script:serverProcess = $null

    Start-Sleep -Seconds 1
    Update-Status
}

# -- Status check ------------------------------------------------------------

function Update-Status {
    $script:serverRunning = Test-Endpoint "$PRE_URL/api/status"
    $script:ollamaRunning = Test-Endpoint "$OLLAMA_URL/v1/models"

    # Update icon
    $notifyIcon.Icon = New-TrayIcon -Running $script:serverRunning

    # Update tooltip
    if ($script:serverRunning) {
        $notifyIcon.Text = "PRE - Running on port $WEB_PORT"
    } else {
        $notifyIcon.Text = "PRE - Server stopped"
    }

    # Update menu items
    if ($script:serverRunning) {
        $statusItem.Text = "Server: running on port $WEB_PORT"
        $startItem.Visible = $false
        $stopItem.Visible = $true
        $restartItem.Enabled = $true
    } else {
        $statusItem.Text = "Server: stopped"
        $startItem.Visible = $true
        $stopItem.Visible = $false
        $restartItem.Enabled = $false
    }

    if ($script:ollamaRunning) {
        $ollamaItem.Text = "Ollama: running"
    } else {
        $ollamaItem.Text = "Ollama: not running"
    }
}

# -- Timer for periodic status checks ----------------------------------------

$script:timer = New-Object System.Windows.Forms.Timer
$script:timer.Interval = $CHECK_INTERVAL_MS
$script:timer.Add_Tick({ Update-Status })
$script:timer.Start()

# Initial check
Update-Status

# -- Run the message loop ----------------------------------------------------

[System.Windows.Forms.Application]::Run()
