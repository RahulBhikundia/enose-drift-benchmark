@echo off
rem Waits for the selection study, then runs the swa_full final runs (UCI + SmellNet) and re-analyses.
:wait
findstr /c:"ALLDONE" results\selection_study.log >nul 2>&1 && goto go
timeout /t 60 /nobreak >nul
goto wait
:go
.venv\Scripts\python.exe scripts\experiments\run_final.py --datasets uci smell --methods swa_full > results\final_swafull.log 2>&1
echo ALLDONE >> results\final_swafull.log
