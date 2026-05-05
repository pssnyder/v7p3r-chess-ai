@echo off
REM Quick Test - Preprocess 1000 puzzles and train for 5 epochs
REM Estimated time: 15-30 minutes

echo ========================================
echo V7P3RAI v4.0 - Quick Start Test
echo ========================================
echo.
echo This will:
echo 1. Preprocess 1,000 puzzles (~5-10 mins)
echo 2. Train model for 5 epochs (~10-15 mins)
echo 3. Validate results
echo.
echo Total time: 15-30 minutes
echo.
pause

cd "e:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v4.0"

python scripts/quick_start_training.py

echo.
echo ========================================
echo Quick Start Complete!
echo ========================================
echo.
echo If successful, launch full preprocessing with:
echo START_FULL_PREPROCESSING.bat
echo.
pause
