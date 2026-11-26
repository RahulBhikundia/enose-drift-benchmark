@echo off
rem Waits for the few-shot runs to finish (keeps at most 2 GPU jobs at once), then the Twin pipeline:
rem tuning -> SA-MSAM tuning -> frozen final runs (10 seeds) -> RF -> analysis -> stats. Resumable.
:wait
findstr /c:"DONE fewshot" results\fewshot_final.log >nul 2>&1 && goto go
timeout /t 60 /nobreak >nul
goto wait
:go
.venv\Scripts\python.exe scripts\experiments\run_twin.py --phase tune >> results\twin_pipeline.log 2>&1
.venv\Scripts\python.exe scripts\experiments\run_twin.py --phase tune2 >> results\twin_pipeline.log 2>&1
findstr /c:"DONE twin tune2" results\twin_pipeline.log >nul 2>&1 || exit /b 1
.venv\Scripts\python.exe scripts\experiments\run_final.py --datasets twin twinreg --seeds 0 1 2 3 4 5 6 7 8 9 --methods sgd adamw sam msam sagm disam coral groupdro erm_mixed domnorm ccpca driftaug swa samsam_disagree samsam_random >> results\twin_pipeline.log 2>&1
.venv\Scripts\python.exe scripts\experiments\baseline_rf.py >> results\twin_pipeline.log 2>&1
.venv\Scripts\python.exe scripts\analysis\analyze_final.py > results\analysis\analysis_full.txt 2>&1
.venv\Scripts\python.exe scripts\analysis\stats_tests.py > results\analysis\stats_full.txt 2>&1
echo ALLDONE >> results\twin_pipeline.log
