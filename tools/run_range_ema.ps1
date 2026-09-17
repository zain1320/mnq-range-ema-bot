param(
    [int]$RestartDelaySec = 30,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$BotRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $BotRoot "runtime\logs"
$PidFile = Join-Path $BotRoot "runtime\range_ema.pid"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

while ($true) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $stdout = Join-Path $LogDir "range_ema_$stamp.log"
    $stderr = Join-Path $LogDir "range_ema_$stamp.err.log"
    $process = Start-Process -FilePath "python" -ArgumentList @(
        "-u", (Join-Path $BotRoot "range_ema_live.py")
    ) -PassThru -NoNewWindow `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr
    Set-Content -Path $PidFile -Value $process.Id
    Write-Host "[range-ema] child PID=$($process.Id) log=$stdout"
    $process.WaitForExit()
    Remove-Item $PidFile -ErrorAction SilentlyContinue
    if (Test-Path $stderr) {
        Get-Content $stderr | Add-Content $stdout
    }
    if ($Once -or $process.ExitCode -eq 0) {
        break
    }
    Write-Host(
        "[range-ema] exit=$($process.ExitCode); restarting in ${RestartDelaySec}s"
    )
    Start-Sleep -Seconds $RestartDelaySec
}
