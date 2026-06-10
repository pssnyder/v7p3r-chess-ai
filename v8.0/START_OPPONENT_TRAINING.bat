@echo off
REM V8.0 Opponent-Based Training Launcher
REM
REM This script launches the enhanced v8.0 training with opponent diversity
REM
REM Expected outcomes:
REM - 20 generations, ~100 games each (2000+ total games)
REM - ~3-4 hour training duration
REM - Win rates tracked vs each opponent
REM - Learned feature weights evolving
REM
REM Goals:
REM - Beat all v7p3r versions from v18+ backwards
REM - Reach tablebase positions in 20-30 moves
REM - Break out of mobility-only focus

echo ============================================================
echo V7P3R v8.0 - OPPONENT-BASED TRAINING
echo ============================================================
echo.
echo Configuration:
echo   Generations: 20
echo   Games/Gen: 100
echo   Batch Size: 512
echo   Total Games: ~2000
echo.
echo Opponents:
echo   - Random Opponent (baseline)
echo   - Material Opponent v2.0 (tactical)
echo   - Positional Opponent v2.0 (strategic)
echo   - V7P3R v17.1 (balanced, ELO 1700)
echo   - V7P3R v17.8 (aggressive, ELO 1800)
echo   - V7P3R v18.3 (balanced, ELO 1850)
echo.
echo Expected Duration: 3-4 hours
echo.
echo ============================================================
pause

cd /d "%~dp0src"

echo.
echo Starting training...
echo.

python train_v8_opponents.py

echo.
echo ============================================================
echo Training complete! Check ../training/v8_opponent_training/
echo ============================================================
pause
