@echo off
rem Waits for the Twin pipeline AND the UCI/SmellNet swa_full runs, then runs swa_full on Twin (10 seeds)
rem and re-runs the full analysis + stats so every dataset includes swa_full.
:wait
findstr /c:"ALLDONE" results\twin_pipeline.log >nul 2>&1 || goto sleep
findstr /c:"ALLDONE" results\final_swafull.log >nul 2>&1 && goto go
:sleep
timeout /t 60 /nobreak >nul
goto wait
:go
.venv\Scripts\python.exe scripts\experiments\run_final.py --datasets twin twinreg --seeds 0 1 2 3 4 5 6 7 8 9 --methods swa_full > results\final_swatwin.log 2>&1
.venv\Scripts\python.exe scripts\analysis\analyze_final.py > results\analysis\analysis_full.txt 2>&1
.venv\Scripts\python.exe scripts\analysis\stats_tests.py > results\analysis\stats_full.txt 2>&1
echo ALLDONE >> results\final_swatwin.log
