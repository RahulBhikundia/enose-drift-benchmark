@echo off
rem Stage 4 tuning -> selection -> validation comparison -> frozen final runs -> analysis. Resumable.
.venv\Scripts\python.exe scripts\experiments\run_grid.py --stage 4 > results\tuning_stage4.log 2>&1
.venv\Scripts\python.exe scripts\experiments\select_hp.py --stage 4 >> results\tuning_stage4.log 2>&1
if not exist results\tuning\selected_stage4.json exit /b 1
.venv\Scripts\python.exe scripts\experiments\go_nogo.py > results\analysis\validation_all_methods.txt 2>&1
.venv\Scripts\python.exe scripts\experiments\run_final.py --methods ccpca driftaug swa > results\final_stage4.log 2>&1
.venv\Scripts\python.exe scripts\analysis\analyze_final.py > results\analysis\analysis_full.txt 2>&1
echo ALLDONE >> results\final_stage4.log
