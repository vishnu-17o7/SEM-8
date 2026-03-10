# run-qkd.ps1 — Launch QKD backend + frontend servers
# Usage: .\run-qkd.ps1

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

$BACKEND_PORT = 8001
$FRONTEND_PORT = 8787

Write-Host "`n=== QKD Dashboard Launcher ===" -ForegroundColor Cyan
Write-Host "  Backend  : http://localhost:$BACKEND_PORT"
Write-Host "  Frontend : http://localhost:$FRONTEND_PORT/qkd-dashboard.html`n"

# Start backend (inference server)
Write-Host "[1/2] Starting backend server..." -ForegroundColor Yellow
$backend = Start-Process -FilePath "python" `
    -ArgumentList "research\qkd_backend.py" `
    -WorkingDirectory $ROOT `
    -PassThru -NoNewWindow

Start-Sleep -Seconds 2

# Start frontend (static file server)
Write-Host "[2/2] Starting frontend server..." -ForegroundColor Yellow
$frontend = Start-Process -FilePath "python" `
    -ArgumentList "-m http.server $FRONTEND_PORT --directory research" `
    -WorkingDirectory $ROOT `
    -PassThru -NoNewWindow

Write-Host "`nBoth servers running. Press Ctrl+C to stop.`n" -ForegroundColor Green

# Wait for Ctrl+C, then clean up both processes
try {
    while ($true) { Start-Sleep -Seconds 1 }
}
finally {
    Write-Host "`nShutting down..." -ForegroundColor Red
    if (!$backend.HasExited)  { Stop-Process -Id $backend.Id  -Force -ErrorAction SilentlyContinue }
    if (!$frontend.HasExited) { Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue }
    Write-Host "Done." -ForegroundColor Green
}
