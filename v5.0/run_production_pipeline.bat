@echo off
REM V7P3R AI v5.0 - Training Pipeline PRODUCTION RUN
REM ================================================
REM Full pipeline: All games, standard features, depth 20
REM Expected time: 2-3 hours
REM Expected output: ~100,000 positions

echo ================================================
echo V7P3R AI v5.0 - TRAINING PIPELINE PRODUCTION RUN
echo ================================================
echo.
echo Configuration:
echo   - Source: Lichess V7P3R Bot PGNs (ALL GAMES)
echo   - Feature Set: standard
echo   - Stockfish Depth: 20
echo   - Expected Time: 2-3 hours
echo   - Expected Positions: ~100,000
echo.
echo WARNING: This will take several hours!
echo.
echo ================================================
echo.

pause

cd /d "E:\Programming Stuff\Chess Engines\V7P3R Chess AI\v7p3r-chess-ai\v5.0"

REM Create timestamped output directory
set TIMESTAMP=%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set OUTPUT_DIR=data/training/production_%TIMESTAMP%

echo Starting pipeline...
echo Output directory: %OUTPUT_DIR%
echo.

python scripts/run_training_pipeline.py ^
  --pgn-dir "E:\Programming Stuff\Chess Engines\Chess Engine Playground\engine-metrics\raw_data\game_records\Lichess V7P3R Bot" ^
  --output-dir "%OUTPUT_DIR%" ^
  --feature-set standard ^
  --stockfish-path "stockfish" ^
  --stockfish-depth 20 ^
  --stockfish-time-limit 10.0

echo.
echo ================================================
echo PRODUCTION RUN COMPLETE!
echo ================================================
echo.
echo Output location: %OUTPUT_DIR%/stage3_graded/
echo.
echo Next steps:
echo   1. Review pipeline_stats.json
echo   2. Validate dataset quality
echo   3. Build PyTorch dataset loader
echo   4. Train V7P3R AI v5.0 model
echo.

pause
