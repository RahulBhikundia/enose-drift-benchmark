# TTA 5-seed watchdog (2026-11-14). Every 5 min: if the chain (run_tta.py final seeds 0-4 -> report) is unfinished
# and not running, restart it (final mode resumes: finished (dataset, split, seed) rows are skipped). Max 3 restarts.
# Log: results\watchdogtta.log
Set-Location (Split-Path $PSScriptRoot -Parent)
$log = "results\watchdogtta.log"; $restarts = 0
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content $log }
function Running($pat) { [bool](Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like $pat -and $_.ProcessId -ne $PID }) }
Log "watchdogtta started (PID $PID)"
while ($true) {
    $done = (Test-Path results\tta_final5.log) -and (Select-String -Path results\tta_final5.log -Pattern ALLDONE -Quiet)
    if ($done) { Log "ALLDONE seen; watchdogtta exiting"; break }
    if (-not (Running '*run_tta.py*')) {
        if ($restarts -lt 3) {
            $restarts++; Log "TTA chain not running -> restart #$restarts"
            Start-Process cmd.exe -ArgumentList '/c', '.venv\Scripts\python.exe scripts\experiments\run_tta.py --mode final --seeds 0 1 2 3 4 >> results\tta_final5.log 2>&1 && .venv\Scripts\python.exe scripts\experiments\run_tta.py --mode report > results\tta\report_final.txt 2>&1 && echo ALLDONE >> results\tta_final5.log' -WindowStyle Hidden
        } else { Log "restart limit reached" }
    }
    Start-Sleep 300
}
