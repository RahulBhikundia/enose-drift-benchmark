# Selection-study watchdog (2026-11-12). Every 5 min: if the selection chain (tuning -> selection -> final runs ->
# analysis) is unfinished and not running, restart it (all steps skip finished runs). Max 3 restarts.
# Log: results\watchdogsel.log
Set-Location (Split-Path $PSScriptRoot -Parent)
$log = "results\watchdogsel.log"; $restarts = 0
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content $log }
function Running($pat) { [bool](Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like $pat -and $_.ProcessId -ne $PID }) }
Log "watchdogsel started (PID $PID)"
while ($true) {
    $done = (Test-Path results\selection_study.log) -and (Select-String -Path results\selection_study.log -Pattern ALLDONE -Quiet)
    if ($done) { Log "ALLDONE seen; watchdogsel exiting"; break }
    if (-not (Running '*after_selection.cmd*')) {
        if ($restarts -lt 3) {
            $restarts++; Log "selection chain not running -> start/restart #$restarts"
            Start-Process cmd.exe -ArgumentList '/c', 'scripts\orchestration\after_selection.cmd' -WindowStyle Hidden
        } else { Log "restart limit reached" }
    }
    Start-Sleep 300
}
