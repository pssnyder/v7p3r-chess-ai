@echo off
REM V7P3R v18.3.1 Profiling Engine Launcher
REM Consolidated 3-file architecture for evaluation profiling

echo ============================================================
echo V7P3R v18.3.1 Profiling Engine
echo ============================================================
echo Consolidated architecture:
echo   - v7p3r_engine.py (search + UCI + openings)
echo   - v7p3r_evaluators.py (all 58+ evaluation functions)
echo   - v7p3r_profiler.py (BigQuery data collection)
echo ============================================================
echo.

py -3 "%~dp0src\v7p3r_engine.py"
