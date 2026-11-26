# Stage-4 watchdog (2026-11-12). Every 5 min: if the stage-4 chain (tuning -> selection -> final runs ->
# analysis) is unfinished and not running, restart it (all steps skip finished runs). Max 3 restarts.
# Log: results\watchdog4.log
Set-Location (Split-Path $PSScriptRoot -Parent)
$log = "results\watchdog4.log"; $restarts = 0
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content $log }
function Running($pat) { [bool](Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like $pat -and $_.ProcessId -ne $PID }) }
Log "watchdog4 started (PID $PID)"
while ($true) {
    $done = (Test-Path results\final_stage4.log) -and (Select-String -Path results\final_stage4.log -Pattern ALLDONE -Quiet)
    if ($done) { Log "ALLDONE seen; watchdog4 exiting"; break }
    if (-not (Running '*after_stage4.cmd*')) {
        if ($restarts -lt 3) {
            $restarts++; Log "stage-4 chain not running -> start/restart #$restarts"
            Start-Process cmd.exe -ArgumentList '/c', 'scripts\orchestration\after_stage4.cmd' -WindowStyle Hidden
        } else { Log "restart limit reached" }
    }
    Start-Sleep 300
}
