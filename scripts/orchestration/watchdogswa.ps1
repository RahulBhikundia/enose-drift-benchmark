# swa_full watchdog (2026-11-12). Every 5 min: if the swa_full chain (tuning -> selection -> final runs ->
# analysis) is unfinished and not running, restart it (all steps skip finished runs). Max 3 restarts.
# Log: results\watchdogswa.log
Set-Location (Split-Path $PSScriptRoot -Parent)
$log = "results\watchdogswa.log"; $restarts = 0
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content $log }
function Running($pat) { [bool](Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like $pat -and $_.ProcessId -ne $PID }) }
Log "watchdogswa started (PID $PID)"
while ($true) {
    $done = (Test-Path results\final_swafull.log) -and (Select-String -Path results\final_swafull.log -Pattern ALLDONE -Quiet)
    if ($done) { Log "ALLDONE seen; watchdogswa exiting"; break }
    if (-not (Running '*after_swafull.cmd*')) {
        if ($restarts -lt 3) {
            $restarts++; Log "swa_full chain not running -> start/restart #$restarts"
            Start-Process cmd.exe -ArgumentList '/c', 'scripts\orchestration\after_swafull.cmd' -WindowStyle Hidden
        } else { Log "restart limit reached" }
    }
    Start-Sleep 300
}
