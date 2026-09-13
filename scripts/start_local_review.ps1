<#
.SYNOPSIS
    One-command local startup for the on-demand Code Review Agent flow.

.DESCRIPTION
    Starts PostgreSQL (docker compose) if it is not already healthy, starts
    the Code Review Agent and waits for /health, starts the Smee webhook
    relay, and prints a READY summary with the URL to paste into the GitHub
    App's Webhook URL field.

    This script never starts a review. Opening or updating a PR only
    creates/resets its "DataNet Code Review" Check Run; a review only starts
    when you click that Check Run's "Run Code Review" action yourself.

    All environment-specific values (model service host, database URL, the
    Smee channel) come from -EnvFile (default .env) - nothing here is
    hardcoded, so the same script and application work unchanged against a
    different deployment's .env (e.g. a shared internal server) by pointing at
    that environment's own values.

.PARAMETER EnvFile
    Path to the env file to load (default: .env next to this script's
    parent directory - the service root). Lets this script be dry-run
    against a scratch env file without touching a real .env.
#>

param(
    [string]$EnvFile
)

$ErrorActionPreference = "Stop"

$ServiceRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ServiceRoot

if (-not $EnvFile) {
    $EnvFile = Join-Path $ServiceRoot ".env"
}

$LogDir = Join-Path $ServiceRoot "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

# ---- Load the env file into this process's environment. -------------------
# Never overrides a variable already set in the shell, so an operator can
# still override any single value ad hoc without editing the file.
if (-not (Test-Path $EnvFile)) {
    Write-Error "Env file not found at $EnvFile. Copy .env.example to .env and fill it in first."
    exit 1
}
Get-Content $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) { return }
    if ($line -match '^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
        $name = $matches[1]
        $value = $matches[2].Trim().Trim('"').Trim("'")
        if (-not (Test-Path "env:$name")) {
            [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

if (-not $env:SMEE_URL) {
    Write-Error "SMEE_URL is not set. Create a channel at https://smee.io/new once, then set SMEE_URL=<that URL> in $EnvFile."
    exit 1
}

$Port = if ($env:PORT) { $env:PORT } else { "8000" }
$HealthUrl = "http://127.0.0.1:$Port/health"

# ---- 1. PostgreSQL (docker compose) ---------------------------------------
Write-Host "==> Starting PostgreSQL (docker compose up -d)..."
docker compose up -d
if ($LASTEXITCODE -ne 0) {
    Write-Error "docker compose up failed - see output above."
    exit 1
}

Write-Host "==> Waiting for PostgreSQL to become healthy..."
$pgHealthy = $false
for ($i = 0; $i -lt 60; $i++) {
    $status = docker compose ps postgres --format json 2>$null | ConvertFrom-Json -ErrorAction SilentlyContinue
    if ($status -and $status.Health -eq "healthy") { $pgHealthy = $true; break }
    Start-Sleep -Seconds 1
}
if (-not $pgHealthy) {
    Write-Error "PostgreSQL did not become healthy within 60s. Check: docker compose logs postgres"
    exit 1
}
Write-Host "    PostgreSQL is healthy."

# ---- 2. Code Review Agent ---------------------------------------------------
# Runs the venv's python directly, not "uv run python" - "uv run" spawns
# python as its own child process, so the PID Start-Process hands back would
# be uv's wrapper, not the actual server; stopping only that PID would leave
# the real server running. Requires `uv sync --extra dev` to have been run
# at least once (see README) so the venv already exists.
Write-Host "==> Starting the Code Review Agent..."
$PythonExe = Join-Path $ServiceRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Error "$PythonExe not found. Run 'uv sync --extra dev' once first."
    exit 1
}
$serviceOutLog = Join-Path $LogDir "service.out.log"
$serviceErrLog = Join-Path $LogDir "service.err.log"
$serviceProc = Start-Process -FilePath $PythonExe `
    -ArgumentList "-m", "review_agent.main" `
    -WorkingDirectory $ServiceRoot `
    -RedirectStandardOutput $serviceOutLog `
    -RedirectStandardError $serviceErrLog `
    -WindowStyle Hidden -PassThru
Set-Content -Path (Join-Path $LogDir "service.pid") -Value $serviceProc.Id

Write-Host "==> Waiting for $HealthUrl ..."
$serviceHealthy = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        $resp = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2
        if ($resp.status -eq "ok") { $serviceHealthy = $true; break }
    } catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $serviceHealthy) {
    Write-Error "Service did not become healthy within 60s. Check: $serviceErrLog"
    exit 1
}
Write-Host "    Service is healthy."

# ---- 3. Webhook relay (Smee) ------------------------------------------------
# Smee, not ngrok: a channel URL (https://smee.io/<id>) is a fixed address
# you create once and reuse - ngrok's free tier hands out a new random URL
# every time it restarts, which would mean re-entering the GitHub App's
# Webhook URL after every local restart. Configurable via SMEE_URL, never
# hardcoded, so a different relay/URL is a .env change, not a code change.
Write-Host "==> Starting the Smee webhook relay..."
$smeeOutLog = Join-Path $LogDir "smee.out.log"
$smeeErrLog = Join-Path $LogDir "smee.err.log"
$smeeProc = Start-Process -FilePath "npx.cmd" `
    -ArgumentList "--yes", "smee-client", "--url", $env:SMEE_URL, "--path", "/webhooks/github", "--port", $Port `
    -WorkingDirectory $ServiceRoot `
    -RedirectStandardOutput $smeeOutLog `
    -RedirectStandardError $smeeErrLog `
    -WindowStyle Hidden -PassThru
Set-Content -Path (Join-Path $LogDir "smee.pid") -Value $smeeProc.Id
Start-Sleep -Seconds 2

# ---- READY ------------------------------------------------------------------
Write-Host ""
Write-Host "================ READY ================="
Write-Host " Code Review Agent : $HealthUrl"
Write-Host " Webhook relay      : $($env:SMEE_URL)"
Write-Host "                      -> http://127.0.0.1:$Port/webhooks/github"
Write-Host ""
Write-Host " Set this as the GitHub App's Webhook URL:"
Write-Host "   $($env:SMEE_URL)"
Write-Host ""
Write-Host " Service PID: $($serviceProc.Id)   Smee PID: $($smeeProc.Id)"
Write-Host " Logs: $LogDir\service.out.log , $LogDir\smee.out.log (and .err.log)"
Write-Host " Stop both:  .\scripts\stop_local_review.ps1"
Write-Host "=========================================="
