@echo off
rem Waits for stage 4 (ALLDONE in results\final_stage4.log), then runs the selection study. Resumable.
:wait
findstr /c:"ALLDONE" results\final_stage4.log >nul 2>&1 && goto go
timeout /t 60 /nobreak >nul
goto wait
:go
.venv\Scripts\python.exe scripts\experiments\run_selection_study.py >> results\selection_study.log 2>&1
findstr /c:"DONE selection" results\selection_study.log >nul 2>&1 && echo ALLDONE >> results\selection_study.log
