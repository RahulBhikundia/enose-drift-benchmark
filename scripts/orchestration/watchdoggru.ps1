# GRU (smellgru) watchdog (2026-11-14). Every 5 min: if the chain (run_smellfeat.py smellgru) is unfinished
# and not running, restart it (runner skips finished result files). Max 3 restarts.
# Log: results\watchdoggru.log
Set-Location (Split-Path $PSScriptRoot -Parent)
$log = "results\watchdoggru.log"; $restarts = 0
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content $log }
function Running($pat) { [bool](Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like $pat -and $_.ProcessId -ne $PID }) }
Log "watchdoggru started (PID $PID)"
while ($true) {
    $done = (Test-Path results\gru.log) -and (Select-String -Path results\gru.log -Pattern ALLDONE -Quiet)
    if ($done) { Log "ALLDONE seen; watchdoggru exiting"; break }
    if (-not (Running '*run_smellfeat.py*smellgru*')) {
        if ($restarts -lt 4) {
            $restarts++; Log "GRU chain not running -> restart #$restarts"
            Start-Process cmd.exe -ArgumentList '/c', '.venv\Scripts\python.exe scripts\experiments\run_smellfeat.py smellgru >> results\gru.log 2>&1 && echo ALLDONE >> results\gru.log' -WindowStyle Hidden
        } else { Log "restart limit reached" }
    }
    Start-Sleep 300
}
