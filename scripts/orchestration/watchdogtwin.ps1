# Twin-pipeline watchdog (2026-11-12). Every 5 min: if the twin chain (tuning -> selection -> final runs ->
# analysis) is unfinished and not running, restart it (all steps skip finished runs). Max 3 restarts.
# Log: results\watchdogtwin.log
Set-Location (Split-Path $PSScriptRoot -Parent)
$log = "results\watchdogtwin.log"; $restarts = 0
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content $log }
function Running($pat) { [bool](Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like $pat -and $_.ProcessId -ne $PID }) }
Log "watchdogtwin started (PID $PID)"
while ($true) {
    $done = (Test-Path results\twin_pipeline.log) -and (Select-String -Path results\twin_pipeline.log -Pattern ALLDONE -Quiet)
    if ($done) { Log "ALLDONE seen; watchdogtwin exiting"; break }
    if (-not (Running '*after_twin.cmd*')) {
        if ($restarts -lt 3) {
            $restarts++; Log "twin chain not running -> start/restart #$restarts"
            Start-Process cmd.exe -ArgumentList '/c', 'scripts\orchestration\after_twin.cmd' -WindowStyle Hidden
        } else { Log "restart limit reached" }
    }
    Start-Sleep 300
}
