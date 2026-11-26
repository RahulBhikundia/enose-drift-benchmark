@echo off
rem Waits for stage-3 tuning + selection, then: validation comparison, frozen final test runs for the
rem new methods, RF baseline, full analysis. Every step is resumable (finished runs are skipped).
:wait
if exist results\tuning\selected_stage3.json goto go
timeout /t 60 /nobreak >nul
goto wait
:go
if not exist results\analysis mkdir results\analysis
.venv\Scripts\python.exe scripts\experiments\go_nogo.py > results\analysis\validation_all_methods.txt 2>&1
.venv\Scripts\python.exe scripts\experiments\run_final.py --methods domnorm erm_mixed sagm disam coral groupdro > results\final_stage3.log 2>&1
.venv\Scripts\python.exe scripts\experiments\baseline_rf.py > results\final_rf.log 2>&1
.venv\Scripts\python.exe scripts\analysis\analyze_final.py > results\analysis\analysis_full.txt 2>&1
echo ALLDONE >> results\final_stage3.log
