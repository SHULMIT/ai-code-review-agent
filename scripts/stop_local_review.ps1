<#
.SYNOPSIS
    Stops the Code Review Agent and Smee relay started by start_local_review.ps1.

.DESCRIPTION
    Both are launched as native processes with their own child processes
    (uv/python's venv shim re-execs the real interpreter; smee-client runs
    under a node wrapper) - a plain Stop-Process on the recorded PID leaves
    the actual server/relay running underneath it. taskkill /T kills the
    whole process tree instead. Does not touch PostgreSQL (docker compose) -
    stop that separately with "docker compose down" if desired.
#>

$ServiceRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $ServiceRoot "logs"

function Stop-Tracked($label, $pidFile) {
    $pidPath = Join-Path $LogDir $pidFile
    if (-not (Test-Path $pidPath)) {
        Write-Host "$label - no PID file at $pidPath, nothing to stop."
        return
    }
    $trackedPid = Get-Content $pidPath -Raw | ForEach-Object { $_.Trim() }
    if (-not $trackedPid) {
        Write-Host "$label - PID file is empty."
        return
    }
    if (-not (Get-Process -Id $trackedPid -ErrorAction SilentlyContinue)) {
        Write-Host "$label (PID $trackedPid) - already stopped."
        Remove-Item $pidPath -ErrorAction SilentlyContinue
        return
    }
    taskkill /F /T /PID $trackedPid | Out-Null
    Write-Host "$label (PID $trackedPid) - stopped."
    Remove-Item $pidPath -ErrorAction SilentlyContinue
}

Stop-Tracked "Code Review Agent" "service.pid"
Stop-Tracked "Smee relay" "smee.pid"
