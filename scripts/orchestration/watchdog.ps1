# Overnight watchdog (2026-11-12). Every 5 min: if a pipeline stage is unfinished and its process is gone,
# restart it (all scripts skip finished runs). At most 3 restarts per stage. Log: results\watchdog.log
Set-Location (Split-Path $PSScriptRoot -Parent)
$log = "results\watchdog.log"; $restarts = @{ tune = 0; chain = 0 }
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content $log }
function Running($pat) { [bool](Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like $pat -and $_.ProcessId -ne $PID }) }
Log "watchdog started (PID $PID)"
while ($true) {
    $done = (Test-Path results\final_stage3.log) -and (Select-String -Path results\final_stage3.log -Pattern ALLDONE -Quiet)
    if ($done) { Log "ALLDONE seen; watchdog exiting"; break }
    if (-not (Test-Path results\tuning\selected_stage3.json) -and -not (Running '*run_grid.py --stage 3*') -and -not (Running '*select_hp.py --stage 3*')) {
        if ($restarts.tune -lt 3) {
            $restarts.tune++; Log "tuning not running and not finished -> restart #$($restarts.tune)"
            Start-Process cmd.exe -ArgumentList '/c', '.venv\Scripts\python.exe scripts\experiments\run_grid.py --stage 3 >> results\tuning_stage3.log 2>&1 && .venv\Scripts\python.exe scripts\experiments\select_hp.py --stage 3 >> results\tuning_stage3.log 2>&1' -WindowStyle Hidden
        } else { Log "tuning: restart limit reached" }
    }
    if (-not (Running '*after_stage3.cmd*')) {
        if ($restarts.chain -lt 3) {
            $restarts.chain++; Log "follow-up chain not running -> restart #$($restarts.chain)"
            Start-Process cmd.exe -ArgumentList '/c', 'scripts\orchestration\after_stage3.cmd' -WindowStyle Hidden
        } else { Log "chain: restart limit reached" }
    }
    Start-Sleep 300
}
